cargo_component_bindings::generate!();

use bindings::component::uq_process::types::*;
use bindings::{
    get_payload, send_requests, send_request, send_response, send_and_await_response, print_to_terminal, receive, Guest,
};
use serde::{Deserialize};
use serde_json::json;
use std::collections::HashMap;

#[allow(dead_code)]
mod process_lib;

struct Component;

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
        print_to_terminal(0, "whisper app: start");

        // 1. http bindings
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
                print_to_terminal(0, "whisper app: got network error");
                continue;
            };
            print_to_terminal(0, "whisper app: got message");
            let Message::Request(request) = message else {
                print_to_terminal(0, "whisper app: got unexpected Response");
                continue;
            };
            print_to_terminal(0, "whisper app: got request");

            if source.process.to_string() == "http_server:sys:uqbar" {
                print_to_terminal(0, "whisper app: got http request");
                let Ok(json) = serde_json::from_slice::<serde_json::Value>(&request.ipc) else {
                    print_to_terminal(0, "whisper app: got invalid json");
                    continue;
                };
                print_to_terminal(0, "whisper app: got http request");

                let mut default_headers = HashMap::new();
                default_headers.insert("Content-Type".to_string(), "text/html".to_string());

                let path = json["path"].as_str().unwrap_or("");
                let method = json["method"].as_str().unwrap_or("");

                match path {
                    "/" => {
                        print_to_terminal(0, "whisper app: sending homepage");
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
                        print_to_terminal(0, "whisper app: got audio post");
                        let Some(form) = get_payload() else {
                            print_to_terminal(0, "whisper app: got invalid payload");
                            continue;
                        };
                        match serde_urlencoded::from_bytes::<AudioForm>(&form.bytes) {
                            Ok(parsed_form) => {
                                let Ok(audio_bytes) = base64::decode(parsed_form.audio.clone()) else {
                                    // print_to_terminal(0, "whisper app: got invalid base64");
                                    print_to_terminal(0, "app:whisper: got invalid base64");
                                    continue;
                                };

                                let res = send_and_await_response(
                                    &Address {
                                        node: our.node.clone(),
                                        process: ProcessId::from_str("nn:whisper:drew.uq").unwrap(),
                                    },
                                    &Request {
                                        inherit: false,
                                        expects_response: Some(30), // TODO evaluate
                                        ipc: vec![],
                                        metadata: None,
                                    },
                                    Some(&Payload {
                                        mime: Some("application/octet-stream".to_string()),
                                        bytes: audio_bytes.clone(),
                                    }),
                                );

                                let text = match res {
                                    Ok((src, msg)) => {
                                        let Message::Response(res) = msg else { panic!(); };
                                        res.0.ipc
                                        // String::from_utf8(res.0.ipc).unwrap()
                                    },
                                    Err(_e) => { "error".to_string().as_bytes().to_vec() }
                                };

                                print_to_terminal(0, &format!("whisper app: output: {:?}", text));
        
                                send_http_response(
                                    200,
                                    default_headers.clone(),
                                    text, // serde_json::to_vec(&res).unwrap(),
                                );
                            }
                            Err(e) => print_to_terminal(0, &format!("whisper app: got invalid form: {:?}", e))
                        }
                    }
                    _ => { todo!() }
                }
            }
        }
    }
}
