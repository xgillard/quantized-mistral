use std::{fs::File, io::{stdin, Read}};

use anyhow::Result;
use candle_core::{utils::cuda_is_available, Device, Tensor};
use candle_core::quantized::gguf_file::Content;
//use candle_transformers::models::quantized_llama::{ModelWeights as Model};
//use candle_transformers::models::quantized_mistral::{Config, VarBuilder, Model};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use structopt::StructOpt;
use tokenizers::Tokenizer;

mod qllama;

use crate::qllama::ModelWeights as Model;

/// This program runs your prompt against a quantized llm and yields its answer
#[derive(StructOpt)]
pub struct Args {
    /// The repository of the original (not quantized) model
    #[structopt(short="o", long, default_value="mistralai/Mistral-7B-Instruct-v0.2")] //default_value="lmz/candle-mistral")] //
    original: String,
    /// The repository of the quantized model
    #[structopt(short="r", long, default_value="TheBloke/Mistral-7B-Instruct-v0.2-GGUF")]//default_value="lmz/candle-mistral")]//
    quantized_repo: String,
    /// The file (specific version) of the quantized model to run
    #[structopt(short="m", long, default_value="mistral-7b-instruct-v0.2.Q5_K_M.gguf")]//default_value="mistral-7b-instruct-v0.2.Q4_K_M.gguf")]//default_value="model-q4k.gguf")]//
    quantized_model: String,
    /// The EOS tokent to use with this model
    #[structopt(short="e", long, default_value="</s>")]
    eos_token: String,
    /// The file whose content is to be used as prompt. If no file is specified, then the prompt is going to be read from stdin
    prompt: Option<String>
}

fn main() -> Result<()>{
    let args = Args::from_args();

    candle_core::quantized::cuda::set_force_dmmv(true);
    candle_core::cuda::set_gemm_reduced_precision_f16(true);
    candle_core::cuda::set_gemm_reduced_precision_bf16(true);

    let device = if cuda_is_available() {
        Device::new_cuda(0)? 
    } else { 
        Device::Cpu 
    };

    let tokenizer = tokenizer(&args.original)?;
    let mut model = model(&args.quantized_repo, &args.quantized_model, &device)?;
    let prompt = prompt(&args.prompt)?;
    
    let response = pipeline(&mut model, &tokenizer, &device, &prompt, &args.eos_token)?;
    println!("{response}");
    Ok(())
}


fn tokenizer(original_repo: &str) -> Result<Tokenizer> {
    let hub_api = hf_hub::api::sync::Api::new()?;
    let hub_api = hub_api.model(original_repo.to_string());
    let tokenizer_file =  hub_api.get("tokenizer.json")?;

    let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(anyhow::Error::msg)?;
    Ok(tokenizer)
}

fn model(quantized_repo: &str, file_id: &str, device: &Device) -> Result<Model> {
    let hub_api = hf_hub::api::sync::Api::new()?;
    let hub_api = hub_api.model(quantized_repo.to_string());

    let model_file = hub_api.get(file_id)?;

    // THIS  WOULD HAVE BEEN THE 'CLEAN WAY' TO GO
    // ---------------------------------------------------------------------------------------
    //let flash_attn: bool = false;
    //let config = Config::config_7b_v0_1(flash_attn);
    //let varbuilder = VarBuilder::from_gguf(model_file, device)?;
    //let model = Model::new(&config, varbuilder)?;

    // THIS IS THE MORE GENERIC APPROACH THAT CAN BE USED WIH THEBLOKE'S QUANTIZED MODELS
    // using quantized llama as a basis
    let mut model_file = File::open(model_file)?;
    let content = Content::read(&mut model_file)?;
    let model = Model::from_gguf(content, &mut model_file, device)?;
    Ok(model)
}

fn prompt(source: &Option<String>) -> Result<String> {
    if let Some(file) = source {
        let mut source = File::open(file)?;
        let mut string = String::new();
        source.read_to_string(&mut string)?;
        Ok(string)
    } else {
        let mut string = String::new();
        stdin().read_to_string(&mut string)?;
        Ok(string)
    }
}

fn pipeline(model: &mut Model, tokenizer: &Tokenizer, device: &Device, prompt: &str, eos_token: &str) -> Result<String> {
    let add_special_tokens = true;
    let seed = 42;
    
    let eos_token_id = tokenizer.get_vocab(true).get(eos_token).copied().unwrap();
    let mut processor = LogitsProcessor::from_sampling(seed, Sampling::ArgMax);

    let inputs = tokenizer.encode(prompt, add_special_tokens).map_err(anyhow::Error::msg)?;
    let inputs = inputs.get_ids();
    
    // priming the model with the prompt
    let mut pos = 0;
    let mut next_token = 0;
    for chunk in inputs.chunks(512) {
        let x = Tensor::new(chunk, device)?.unsqueeze(0)?;
        let logits = model.forward_with_mask(&x, pos, None)?.reshape(((),))?;
        pos       += chunk.len();
        next_token = processor.sample(&logits)?;
    }
    println!("== primed ==");
    // actually generate the output (one token at the time)
    let mut tokens = vec![next_token];
    loop { // on ne s'arrete que lorsque le eos token a été produit
        let x = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
        let logits = model.forward(&x, pos)?.reshape(((),))?;
        
        pos       += 1;
        next_token = processor.sample(&logits)?;
        tokens.push(next_token);

        if next_token == eos_token_id { break; }
    }

    let response = tokenizer.decode(&tokens, true).map_err(anyhow::Error::msg)?;
    Ok(response)
}