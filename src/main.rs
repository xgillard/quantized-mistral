use std::fs::File;

use anyhow::Result;
use candle_core::{quantized::gguf_file::Content, utils::cuda_is_available, Device, Tensor};
//use candle_transformers::models::quantized_mistral::{Config, VarBuilder, Model};
use candle_transformers::{generation::{LogitsProcessor, Sampling}, models::quantized_llama::ModelWeights as Model};
use tokenizers::Tokenizer;

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
    //let flash_attn = true;
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

fn pipeline(model: &mut Model, tokenizer: &Tokenizer, device: &Device, prompt: &str, eos_token: &str) -> Result<String> {
    let add_special_tokens = true;
    let seed = 42;
    
    let eos_token_id = tokenizer.get_vocab(true).get(eos_token).copied().unwrap();
    let mut processor = LogitsProcessor::from_sampling(seed, Sampling::ArgMax);

    let inputs = tokenizer.encode(prompt, add_special_tokens).map_err(anyhow::Error::msg)?;
    let prompt_len = inputs.get_ids().len();

    let mut next_token = 0;
    // priming the model with the prompt
    for (pos, token) in inputs.get_ids().iter().copied().enumerate() {
        let x = Tensor::new(&[token], device)?.unsqueeze(0)?;
        let logits = model.forward(&x, pos)?.squeeze(0)?;
        next_token = processor.sample(&logits)?;
    }

    // actually generate the output (one token at the time)
    let mut tokens = vec![];
    for pos in 0.. { // on ne s'arrete que lorsque le eos token a été produit
        let x = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
        let logits = model.forward(&x, prompt_len + pos)?.squeeze(0)?;
        next_token = processor.sample(&logits)?;
        tokens.push(next_token);

        if next_token == eos_token_id { break; }
    }

    let response = tokenizer.decode(&tokens, true).map_err(anyhow::Error::msg)?;
    Ok(response)
}

fn main() -> Result<()>{
    let original = "mistralai/Mistral-7B-Instruct-v0.2";
    let model_id = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF";
    let file_id  = "mistral-7b-instruct-v0.2.Q4_K_M.gguf";
    //
    let eos      = "</s>";

    let device = if cuda_is_available() {
        Device::new_cuda(0)? 
    } else { 
        Device::Cpu 
    };

    let tokenizer = tokenizer(original)?;
    let mut model = model(model_id, file_id, &device)?;

    let prompt = "[INST] Ecris moi une histoire du soir pour deux jeunes garcons (Simon et Augustin) qui ont 8 et 6 ans. Je veux que ton histoire parle de dinosaures, de drones et de montagnes russes a efteling [/INST]";
    let response = pipeline(&mut model, &tokenizer, &device, prompt, eos)?;
    println!("{response}");
    Ok(())
}
