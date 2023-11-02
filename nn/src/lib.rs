cargo_component_bindings::generate!();
use anyhow::Error as E;
use candle::{safetensors::Load, DType, Device, IndexOp, Tensor, D};
use candle_nn::{ops::softmax, VarBuilder};
use candle_transformers::models::whisper::{self as m, Config};
use crate::languages::LANGUAGES;
use rand::{distributions::Distribution, rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use tokenizers::Tokenizer;

mod process_lib;
mod languages;
mod audio;

use bindings::component::uq_process::types::*;
use bindings::{
    get_payload, send_requests, send_response, print_to_terminal, receive, Guest,
};

struct Component;

// loading the model
// NOTE : currently just tiny english model. Can always switch to multilingual or quantized
const MEL_FILTERS: &[u8] = include_bytes!("mel_filters.safetensors");
const TOKENIZER: &[u8] = include_str!("whisper-tiny.en/tokenizer.json").as_bytes();
const WEIGHTS: &[u8] = include_bytes!("whisper-tiny.en/model.safetensors");
const CONFIG: &[u8] = include_str!("whisper-tiny.en/config.json").as_bytes();

pub const DTYPE: DType = DType::F32;

pub enum Model {
    Normal(m::model::Whisper),
    Quantized(m::quantized_model::Whisper),
}

// Maybe we should use some traits rather than doing the dispatch for all these.
impl Model {
    pub fn config(&self) -> &Config {
        match self {
            Self::Normal(m) => &m.config,
            Self::Quantized(m) => &m.config,
        }
    }

    pub fn encoder_forward(&mut self, x: &Tensor, flush: bool) -> candle::Result<Tensor> {
        match self {
            Self::Normal(m) => m.encoder.forward(x, flush),
            Self::Quantized(m) => m.encoder.forward(x, flush),
        }
    }

    pub fn decoder_forward(
        &mut self,
        x: &Tensor,
        xa: &Tensor,
        flush: bool,
    ) -> candle::Result<Tensor> {
        match self {
            Self::Normal(m) => m.decoder.forward(x, xa, flush),
            Self::Quantized(m) => m.decoder.forward(x, xa, flush),
        }
    }

    pub fn decoder_final_linear(&self, x: &Tensor) -> candle::Result<Tensor> {
        match self {
            Self::Normal(m) => m.decoder.final_linear(x),
            Self::Quantized(m) => m.decoder.final_linear(x),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodingResult {
    pub tokens: Vec<u32>,
    pub text: String,
    pub avg_logprob: f64,
    pub no_speech_prob: f64,
    temperature: f64,
    compression_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub start: f64,
    pub duration: f64,
    pub dr: DecodingResult,
}

/// Returns the token id for the selected language.
pub fn detect_language(model: &mut Model, tokenizer: &Tokenizer, mel: &Tensor) -> Result<u32, E> {
    print_to_terminal(0, "detecting language");
    let (_bsize, _, seq_len) = mel.dims3()?;
    let mel = mel.narrow(
        2,
        0,
        usize::min(seq_len, model.config().max_source_positions),
    )?;
    let device = mel.device();

    let language_token_ids = LANGUAGES
        .iter()
        .map(|(t, _)| token_id(tokenizer, &format!("<|{t}|>")))
        .map(|e| e.map_err(E::msg))
        .collect::<Result<Vec<_>, E>>()?;

    let sot_token = token_id(tokenizer, m::SOT_TOKEN)?;
    let audio_features = model.encoder_forward(&mel, true)?;
    let tokens = Tensor::new(&[[sot_token]], device)?;
    let language_token_ids = Tensor::new(language_token_ids.as_slice(), device)?;
    let ys = model.decoder_forward(&tokens, &audio_features, true)?;
    let logits = model.decoder_final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
    let logits = logits.index_select(&language_token_ids, 0)?;
    let probs = candle_nn::ops::softmax(&logits, D::Minus1)?;
    let probs = probs.to_vec1::<f32>()?;
    let mut probs = LANGUAGES.iter().zip(probs.iter()).collect::<Vec<_>>();
    probs.sort_by(|(_, p1), (_, p2)| p2.total_cmp(p1));
    for ((_, language), p) in probs.iter().take(5) {
        println!("{language}: {p}")
    }
    let token = &format!("<|{}|>", probs[0].0 .0);
    let language = token_id(tokenizer, token)?;
    print_to_terminal(0, "detected language: {language} {token}");
    Ok(language)
}
pub fn token_id(tokenizer: &Tokenizer, token: &str) -> candle::Result<u32> {
    match tokenizer.token_to_id(token) {
        None => candle::bail!("no token-id for {token}"),
        Some(id) => Ok(id),
    }
}

pub struct Decoder {
    model: Model,
    rng: rand::rngs::StdRng,
    task: Option<Task>,
    language: Option<String>,
    is_multilingual: bool,
    mel_filters: Vec<f32>,
    timestamps: bool,
    tokenizer: Tokenizer,
    suppress_tokens: Tensor,
    sot_token: u32,
    transcribe_token: u32,
    translate_token: u32,
    eot_token: u32,
    no_speech_token: u32,
    no_timestamps_token: u32,
}

impl Decoder {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        mel_filters: Vec<f32>,
        device: &Device,
        task: Option<Task>,
        language: Option<String>,
        is_multilingual: bool,
        timestamps: bool,
    ) -> anyhow::Result<Self> {
        let suppress_tokens: Vec<f32> = (0..model.config().vocab_size as u32)
            .map(|i| {
                if model.config().suppress_tokens.contains(&i) {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let no_timestamps_token = token_id(&tokenizer, m::NO_TIMESTAMPS_TOKEN)?;
        let suppress_tokens = Tensor::new(suppress_tokens.as_slice(), device)?;
        let sot_token = token_id(&tokenizer, m::SOT_TOKEN)?;
        let transcribe_token = token_id(&tokenizer, m::TRANSCRIBE_TOKEN)?;
        let translate_token = token_id(&tokenizer, m::TRANSLATE_TOKEN)?;
        let eot_token = token_id(&tokenizer, m::EOT_TOKEN)?;
        let no_speech_token = token_id(&tokenizer, m::NO_SPEECH_TOKEN)?;
        let seed = 299792458;
        Ok(Self {
            model,
            rng: StdRng::seed_from_u64(seed),
            tokenizer,
            mel_filters,
            task,
            timestamps,
            language,
            is_multilingual,
            suppress_tokens,
            sot_token,
            transcribe_token,
            translate_token,
            eot_token,
            no_speech_token,
            no_timestamps_token,
        })
    }

    fn decode(&mut self, mel: &Tensor, t: f64) -> anyhow::Result<DecodingResult> {
        let model = &mut self.model;
        let language_token = match (self.is_multilingual, &self.language) {
            (true, None) => Some(detect_language(model, &self.tokenizer, mel)?),
            (false, None) => None,
            (true, Some(language)) => {
                match token_id(&self.tokenizer, &format!("<|{:?}|>", self.language)) {
                    Ok(token_id) => Some(token_id),
                    Err(_) => anyhow::bail!("language {language} is not supported"),
                }
            }
            (false, Some(_)) => {
                anyhow::bail!("a language cannot be set for non-multilingual models")
            }
        };

        let audio_features = model.encoder_forward(mel, true)?;
        println!("audio features: {:?}", audio_features.dims());
        let sample_len = model.config().max_target_positions / 2;
        let mut sum_logprob = 0f64;
        let mut no_speech_prob = f64::NAN;
        let mut tokens = vec![self.sot_token];
        if let Some(language_token) = language_token {
            tokens.push(language_token);
        }
        match self.task {
            None | Some(Task::Transcribe) => tokens.push(self.transcribe_token),
            Some(Task::Translate) => tokens.push(self.translate_token),
        }
        if !self.timestamps {
            tokens.push(self.no_timestamps_token);
        }
        for i in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), mel.device())?;

            // The model expects a batch dim but this inference loop does not handle
            // it so we add it at this point.
            let tokens_t = tokens_t.unsqueeze(0)?;
            let ys = model.decoder_forward(&tokens_t, &audio_features, i == 0)?;

            // Extract the no speech probability on the first iteration by looking at the first
            // token logits and the probability for the according token.
            if i == 0 {
                let logits = model.decoder_final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
                no_speech_prob = softmax(&logits, 0)?
                    .i(self.no_speech_token as usize)?
                    .to_scalar::<f32>()? as f64;
            }

            let (_, seq_len, _) = ys.dims3()?;
            let logits = model
                .decoder_final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?
                .i(0)?;
            // TODO: Besides suppress tokens, we should apply the heuristics from
            // ApplyTimestampRules, i.e.:
            // - Timestamps come in pairs, except before EOT.
            // - Timestamps should be non-decreasing.
            // - If the sum of the probabilities of timestamps is higher than any other tokens,
            //   only consider timestamps when sampling.
            // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L439
            let logits = logits.broadcast_add(&self.suppress_tokens)?;
            let next_token = if t > 0f64 {
                let prs = softmax(&(&logits / t)?, 0)?;
                let logits_v: Vec<f32> = prs.to_vec1()?;
                let distr = rand::distributions::WeightedIndex::new(&logits_v)?;
                distr.sample(&mut self.rng) as u32
            } else {
                let logits_v: Vec<f32> = logits.to_vec1()?;
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(i, _)| i as u32)
                    .unwrap()
            };
            tokens.push(next_token);
            let prob = softmax(&logits, candle::D::Minus1)?
                .i(next_token as usize)?
                .to_scalar::<f32>()? as f64;
            if next_token == self.eot_token || tokens.len() > model.config().max_target_positions {
                break;
            }
            sum_logprob += prob.ln();
        }
        let text = self.tokenizer.decode(&tokens, true).map_err(E::msg)?;
        let avg_logprob = sum_logprob / tokens.len() as f64;

        Ok(DecodingResult {
            tokens,
            text,
            avg_logprob,
            no_speech_prob,
            temperature: t,
            compression_ratio: f64::NAN,
        })
    }

    fn decode_with_fallback(&mut self, segment: &Tensor) -> anyhow::Result<DecodingResult> {
        for (i, &t) in m::TEMPERATURES.iter().enumerate() {
            let dr: Result<DecodingResult, _> = self.decode(segment, t);
            if i == m::TEMPERATURES.len() - 1 {
                return dr;
            }
            // On errors, we try again with a different temperature.
            match dr {
                Ok(dr) => {
                    let needs_fallback = dr.compression_ratio > m::COMPRESSION_RATIO_THRESHOLD
                        || dr.avg_logprob < m::LOGPROB_THRESHOLD;
                    if !needs_fallback || dr.no_speech_prob > m::NO_SPEECH_THRESHOLD {
                        return Ok(dr);
                    }
                }
                Err(err) => {
                    print_to_terminal(0, "Error running at {t}: {err}")
                }
            }
        }
        unreachable!()
    }

    fn run(&mut self, mel: &Tensor) -> anyhow::Result<Vec<Segment>> {
        let (_, _, content_frames) = mel.dims3()?;
        let mut seek = 0;
        let mut segments = vec![];
        while seek < content_frames {
            let time_offset = (seek * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
            let segment_size = usize::min(content_frames - seek, m::N_FRAMES);
            let mel_segment = mel.narrow(2, seek, segment_size)?;
            let segment_duration = (segment_size * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
            let dr = self.decode_with_fallback(&mel_segment)?;
            seek += segment_size;
            if dr.no_speech_prob > m::NO_SPEECH_THRESHOLD && dr.avg_logprob < m::LOGPROB_THRESHOLD {
                print_to_terminal(0, "no speech detected, skipping {seek} {dr:?}");
                continue;
            }
            let segment = Segment {
                start: time_offset,
                duration: segment_duration,
                dr,
            };
            print_to_terminal(0, "{seek}: {segment:?}");
            segments.push(segment)
        }
        Ok(segments)
    }

    // TODO failing compilation for some reason
    pub fn load(md: ModelData) -> anyhow::Result<Self> {
        let device = Device::Cpu;
        let tokenizer = Tokenizer::from_bytes(&md.tokenizer).map_err(E::msg)?;

        let mel_filters = safetensors::tensor::SafeTensors::deserialize(&md.mel_filters)?;
        let mel_filters = mel_filters.tensor("mel_80")?.load(&device)?;
        print_to_terminal(0, &format!("loaded mel filters {:?}", mel_filters.shape()));
        let mel_filters = mel_filters.flatten_all()?.to_vec1::<f32>()?;
        let config: Config = serde_json::from_slice(&md.config)?;
        let model = if md.quantized {
            let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf_buffer(
                &md.weights,
            )?;
            Model::Quantized(m::quantized_model::Whisper::load(&vb, config)?)
        } else {
            let vb = VarBuilder::from_buffered_safetensors(md.weights, m::DTYPE, &device)?;
            Model::Normal(m::model::Whisper::load(&vb, config)?)
        };
        print_to_terminal(0, "done loading model");

        let task = match md.task.as_deref() {
            Some("translate") => Some(Task::Translate),
            _ => Some(Task::Transcribe),
        };

        let decoder = Self::new(
            model,
            tokenizer,
            mel_filters,
            &device,
            task,
            md.language,
            md.is_multilingual,
            md.timestamps,
        )?;
        Ok(decoder)
    }

    pub fn convert_and_run(&mut self, wav_input: &[u8]) -> anyhow::Result<Vec<Segment>> {
        let device = Device::Cpu;
        let mut wav_input = std::io::Cursor::new(wav_input);
        let (header, data) = match wav::read(&mut wav_input) {
            Ok(wav) => wav,
            Err(e) => {
                print_to_terminal(0, &format!("error reading wav: {:?}", e));
                panic!();
            }
        };
        if header.sampling_rate != m::SAMPLE_RATE as u32 {
            print_to_terminal(0, &format("wav file must have a {} sampling rate", m::SAMPLE_RATE));
            panic!();
        }
        let data = data.as_sixteen().expect("expected 16 bit wav file");
        let pcm_data: Vec<_> = data[..data.len() / header.channel_count as usize]
            .iter()
            .map(|v| *v as f32 / 32768.)
            .collect();
        print_to_terminal(0, &format!("pcm data loaded {}", pcm_data.len()));
        let mel = crate::audio::pcm_to_mel(&pcm_data, &self.mel_filters)?;
        let mel_len = mel.len();
        let mel = Tensor::from_vec(mel, (1, m::N_MELS, mel_len / m::N_MELS), &device)?;
        print_to_terminal(0, &format!("loaded mel: {:?}", mel.dims()));
        let segments = self.run(&mel)?;
        Ok(segments)
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub enum Task {
    Transcribe,
    Translate,
}

// Communication to the worker happens through bincode, the model weights and configs are fetched
// on the main thread and transfered via the following structure.
#[derive(Serialize, Deserialize)]
pub struct ModelData {
    pub weights: Vec<u8>,
    pub tokenizer: Vec<u8>,
    pub mel_filters: Vec<u8>,
    pub config: Vec<u8>,
    pub quantized: bool,
    pub timestamps: bool,
    pub is_multilingual: bool,
    pub language: Option<String>,
    pub task: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub enum WorkerInput {
    ModelData(ModelData),
    DecodeTask { wav_bytes: Vec<u8> },
}

#[derive(Deserialize, Debug)]
struct AudioForm {
    audio: String,
}

fn send_http_response(status: u16, headers: HashMap<String, String>, payload_bytes: Vec<u8>) {
    send_response(
        &Response {
            inherit: false,
            ipc: serde_json::json!({
                "status": status,
                "headers": headers,
            })
            .to_string()
            .as_bytes()
            .to_vec(),
            metadata: None,
        },
        Some(&Payload {
            mime: Some("application/octet-stream".to_string()),
            bytes: payload_bytes,
        }),
    )
}

impl Guest for Component {
    fn init(our: Address) {
        print_to_terminal(0, "nn: start");

        // 1. load model data
        let md = ModelData {
            weights: WEIGHTS.to_vec(),
            tokenizer: TOKENIZER.to_vec(),
            mel_filters: MEL_FILTERS.to_vec(),
            config: CONFIG.to_vec(),
            quantized: false,
            timestamps: false,
            is_multilingual: false,
            language: None, // TODO might be "en" or "english"
            task: Some("transcribe".to_string()), // TODO maybe None
        };
        
        let mut decoder = Decoder::load(md).unwrap();

        // 2. http bindings
        let bindings_address = Address {
            node: our.node.clone(),
            process: ProcessId::from_str("http_server:sys:uqbar").unwrap(),
        };
        let http_endpoint_binding_requests: [(Address, Request, Option<Context>, Option<Payload>);
        2] = [
            (
                bindings_address.clone(),
                Request {
                    inherit: false,
                    expects_response: None,
                    ipc: json!({
                        "BindPath": {
                            "path": "/",
                            "authenticated": false, // TODO
                            "local_only": false
                        }
                    })
                    .to_string()
                    .as_bytes()
                    .to_vec(),
                    metadata: None,
                },
                None,
                None,
            ),
            (
                bindings_address.clone(),
                Request {
                    inherit: false,
                    expects_response: None,
                    ipc: json!({
                        "BindPath": {
                            "path": "/audio",
                            "authenticated": false, // TODO
                            "local_only": false
                        }
                    })
                    .to_string()
                    .as_bytes()
                    .to_vec(),
                    metadata: None,
                },
                None,
                None,
            ),
        ];
        send_requests(&http_endpoint_binding_requests);


        loop {
            let Ok((source, message)) = receive() else {
                print_to_terminal(0, "nn: got network error");
                continue;
            };
            print_to_terminal(0, "nn: got message");
            let Message::Request(request) = message else {
                print_to_terminal(0, "nn: got unexpected Response");
                continue;
            };
            print_to_terminal(0, "nn: got request");

            if source.process.to_string() == "http_server:sys:uqbar" {
                print_to_terminal(0, "nn: got http request");
                let Ok(json) = serde_json::from_slice::<serde_json::Value>(&request.ipc) else {
                    print_to_terminal(0, "nn: got invalid json");
                    continue;
                };
                print_to_terminal(0, "nn: got http request");

                let mut default_headers = HashMap::new();
                default_headers.insert("Content-Type".to_string(), "text/html".to_string());

                let path = json["path"].as_str().unwrap_or("");
                let method = json["method"].as_str().unwrap_or("");

                match path {
                    "/" => {
                        print_to_terminal(0, "nn: sending homepage");
                        send_http_response(
                            200,
                            default_headers.clone(),
                            "audio homepage".as_bytes().to_vec(),
                            // CHESS_PAGE
                            //     .replace("${node}", &our.node)
                            //     .replace("${process}", &our.process.to_string())
                            //     .replace("${js}", CHESS_JS)
                            //     .replace("${css}", CHESS_CSS)
                            //     .to_string()
                            //     .as_bytes()
                            //     .to_vec(),
                        );
                    }
                    "/audio" => {
                        print_to_terminal(0, "nn: got audio post");
                        let Some(form) = get_payload() else {
                            print_to_terminal(0, "nn: got invalid payload");
                            continue;
                        };
                        match serde_urlencoded::from_bytes::<AudioForm>(&form.bytes) {
                            Ok(parsed_form) => {
                                let Ok(audio_bytes) = base64::decode(parsed_form.audio.clone()) else {
                                    // print_to_terminal(0, "nn: got invalid base64");
                                    print_to_terminal(0, &format!("nn: got invalid base64: {:?}", parsed_form.audio));
                                    continue;
                                };

                                let output = match decoder
                                    .convert_and_run(&audio_bytes)
                                    {
                                        Ok(output) => output,
                                        Err(e) => vec![]
                                    };
                                
                                let output_texts = output.into_iter().map(|seg| seg.dr.text).collect::<Vec<String>>();
                                
                                print_to_terminal(0, &format!("nn: output: {:?}", output_texts));
        
                                send_http_response(
                                    200,
                                    default_headers.clone(),
                                    serde_json::to_vec(&output_texts).unwrap(),
                                );
                            }
                            Err(e) => print_to_terminal(0, &format!("nn: got invalid form: {:?}", e))
                        }
                    }
                    _ => { todo!() }
                }
            }
        }
    }
}
