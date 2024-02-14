import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset, Audio
import evaluate
import onnxruntime
import numpy as np

DEBUG=True
DEBUG_SAMPLES=32

def get_text(sample):
    if "text" in sample:
        return sample["text"]
    if "sentence" in sample:
        return sample["sentence"]
    if "normalized_text" in sample:
        return sample["normalized_text"]
    if "transcript" in sample:
        return sample["transcript"]
    raise ValueError(f"Sample: {sample.keys()} has no transcript.")

def whisper_norm(whisper_asr, text):
    return  whisper_asr.tokenizer._normalize(text)

def normalize_dataset(dataset, whisper_asr):
    def normalize(batch):
        batch["norm_text"] = whisper_norm(whisper_asr, get_text(batch))
        return batch
    return dataset.map(normalize)

def data(dataset):
    for i, item in enumerate(dataset):
        yield {**item["audio"], "reference": item["norm_text"]}

def is_target_text_in_range(ref):
    if ref.strip() == "ignore time segment in scoring":
        return False
    else:
        return ref.strip() != ""

def load_torch_pipeline(model_id="openai/whisper-large-v2"):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    whisper_asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        #max_new_tokens=128,
        #chunk_length_s=30,
        #batch_size=16,
        #return_timestamps=True,
        torch_dtype=torch_dtype,
        device=0
    )
    return whisper_asr

def test_optimum_pipeline(eval_datasets, model_id="openai/whisper-large-v2", use_cuda=True):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    # optimum-cli export onnx --model openai/whisper-large-v2 whisper-large-v2-with-past/ --task automatic-speech-recognition-with-past --opset 14
    model_path = 'whisper-large-v2-with-past'

    # git clone https://huggingface.co/Intel/whisper-large-v2-onnx-int4
    # model_path = 'whisper-large-v2-onnx-int4'

    from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
    from transformers import PretrainedConfig
    processor = WhisperProcessor.from_pretrained(model_id)
    model_config = PretrainedConfig.from_pretrained(model_id)

    onnx_paths = [os.path.join(model_path, 'encoder_model.onnx'),
                  os.path.join(model_path, 'decoder_model.onnx'),
                  os.path.join(model_path, 'decoder_with_past_model.onnx')]

    device = "cuda" if use_cuda else "cpu"
    sessions = ORTModelForSpeechSeq2Seq.load_model(*onnx_paths, provider='CPUExecutionProvider' if not use_cuda else 'CUDAExecutionProvider')
    model = ORTModelForSpeechSeq2Seq(
        encoder_session = sessions[0],
        decoder_session = sessions[1],
        config = model_config,
        onnx_paths = onnx_paths,
        decoder_with_past_session = sessions[2],
        )
    model.decoder.normalized_config.update({"num_attention_heads":20})
    model.decoder_with_past.normalized_config.update({"num_attention_heads":20})

    wer_results = {}
    wer = evaluate.load("wer")
    for dataset_name, dataset in eval_datasets.items():
        if DEBUG:
            dataset = dataset.take(DEBUG_SAMPLES)

        predictions = []
        references = []

        for i, batch in enumerate(dataset):
            audio = batch["audio"]

            # features = processor.feature_extractor(audio, return_tensors="pt")
            # outputs = model.generate(inputs=features["input_features"])
            # res = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
            if use_cuda:
                input_features = input_features.to(device)
            reference = processor.tokenizer._normalize(get_text(batch))
            references.append(reference)

            predicted_ids = model.generate(input_features)[0]
            transcription = processor.decode(predicted_ids)
            prediction = processor.tokenizer._normalize(transcription)
            predictions.append(prediction)
            print(i, prediction, references)

        wer_result = wer.compute(references=references, predictions=predictions)
        wer_result = round(100 * wer_result, 2)

        wer_results[dataset_name] = wer_result
        print(f"WER for data set {dataset_name}: {wer_result}")

    print(wer_results)


def test_ort_pipeline(eval_datasets, model_id="openai/whisper-large-v2", use_cuda=True, run_torch=False):
    from transformers import WhisperProcessor

    # Load PyTorch model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if run_torch:
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)

    # Load ONNX model that is generated like the following:
    # python3 -m onnxruntime.transformers.models.whisper.convert_to_onnx -m openai/whisper-large-v2 --output whisper-large-v2-beam-search --use_external_data_format --optimize_onnx --precision fp16 --use_gpu --provider cuda --disable_auto_mixed_precision
    model_path = 'whisper-large-v2-beam-search.onnx'
    ort_session = onnxruntime.InferenceSession(model_path, provider='CPUExecutionProvider' if not use_cuda else 'CUDAExecutionProvider')

    from transformers import PretrainedConfig
    processor = WhisperProcessor.from_pretrained(model_id)
    model_config = PretrainedConfig.from_pretrained(model_id)
    device = "cuda" if use_cuda else "cpu"

    wer_results = {}
    wer = evaluate.load("wer")
    for dataset_name, dataset in eval_datasets.items():
        if DEBUG:
            dataset = dataset.take(DEBUG_SAMPLES)

        predictions = []
        references = []

        for i, batch in enumerate(dataset):
            audio = batch["audio"]

            # features = processor.feature_extractor(audio, return_tensors="pt")
            # outputs = model.generate(inputs=features["input_features"])
            # res = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
            if use_cuda:
                input_features = input_features.to('cuda')

            reference = processor.tokenizer._normalize(get_text(batch))
            references.append(reference)

            start_id = [model_config.decoder_start_token_id]  # ex: [50258]
            prompt_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
            prompt_ids = list(map(lambda token: token[1], prompt_ids))  # ex: [50259, 50358, 50363]
            forced_decoder_ids = start_id + prompt_ids  # ex: [50258, 50259, 50358, 50363]

            batch_size, max_length, min_length, num_beams, num_return_sequences = 1, 30, 0, 1, 1
            length_penalty, repetition_penalty = 1.0, 1.0
            inputs = {
                "input_features": input_features.to(device),
                "max_length": max_length,
                "min_length": min_length,
                "num_beams": num_beams,
                "num_return_sequences": num_return_sequences,
                "length_penalty": length_penalty,
                "repetition_penalty": repetition_penalty,
                "early_stopping": True,
                "use_cache": True,
            }

            if run_torch:
                pt_outputs = model.generate(**inputs).detach().cpu().numpy()
                pt_transcription = processor.batch_decode(pt_outputs, skip_special_tokens=True)[0]
                print(pt_transcription)

            del inputs["early_stopping"]
            del inputs["use_cache"]

            ort_names = list(map(lambda entry: entry.name, ort_session.get_inputs()))
            ort_dtypes = list(map(lambda entry: entry.type, ort_session.get_inputs()))
            ort_to_np = {
                "tensor(float)": np.float32,
                "tensor(float16)": np.float16,
                "tensor(int64)": np.int64,
                "tensor(int32)": np.int32,
                "tensor(int8)": np.int8,
                "tensor(uint8)": np.uint8,
            }

            use_extra_decoding_ids = "extra_decoding_ids" in ort_names
            for name, dtype in zip(ort_names, ort_dtypes):
                if name == "input_features":
                    inputs[name] = inputs[name].detach().cpu().numpy()
                elif name == "vocab_mask":
                    inputs[name] = np.ones(model_config.vocab_size, dtype=ort_to_np[dtype])
                elif name == "prefix_vocab_mask":
                    inputs[name] = np.ones((batch_size, model_config.vocab_size), dtype=ort_to_np[dtype])
                elif name == "decoder_input_ids":
                    raw_input_ids = [start_id] if use_extra_decoding_ids else [forced_decoder_ids]
                    inputs[name] = np.array(raw_input_ids, dtype=ort_to_np[dtype])
                elif name == "logits_processor":
                    inputs[name] = np.array([1], dtype=ort_to_np[dtype])
                elif name == "cross_qk_layer_head":
                    inputs[name] = np.array([[0, 0]], dtype=ort_to_np[dtype])
                elif name == "extra_decoding_ids":
                    inputs[name] = np.repeat(np.array([prompt_ids], dtype=ort_to_np[dtype]), batch_size, 0)
                elif name == "temperature":
                    inputs[name] = np.array([1.0], dtype=ort_to_np[dtype])
                else:
                    inputs[name] = np.array([inputs[name]], dtype=ort_to_np[dtype])

            predicted_ids = ort_session.run(None, inputs)[0][0]
            transcription = processor.decode(predicted_ids)
            prediction = processor.tokenizer._normalize(transcription)
            predictions.append(prediction)
            print(i, prediction, references)

        wer_result = wer.compute(references=references, predictions=predictions)
        wer_result = round(100 * wer_result, 2)

        wer_results[dataset_name] = wer_result
        print(f"WER for data set {dataset_name}: {wer_result}")

    print(wer_results)


def load_datasets():
    voxpopuli = load_dataset("facebook/voxpopuli", "en", split="test", streaming=True, trust_remote_code=True)
    if DEBUG:
        return {"VoxPopuli": voxpopuli}

    librispeech_clean = load_dataset("librispeech_asr", "all", split="test.clean", streaming=True,
                                     trust_remote_code=True, cache_dir="cache")

    librispeech_other = load_dataset("librispeech_asr", "all", split="test.other", streaming=True,
                                     trust_remote_code=True)

    gigaspeech = load_dataset("speechcolab/gigaspeech", "xs", split="test", streaming=True, use_auth_token=True,
                              trust_remote_code=True)
    common_voice = load_dataset("mozilla-foundation/common_voice_11_0", "en", revision="streaming", split="test", streaming=True, use_auth_token=True,
                                trust_remote_code=True)

    spgispeech = load_dataset("kensho/spgispeech", "S", split="test", streaming=True, use_auth_token=True,
                              trust_remote_code=True)


    voxpopuli = load_dataset("facebook/voxpopuli", "en", split="test", streaming=True, trust_remote_code=True)
    tedlium = load_dataset("LIUM/tedlium", "release3", split="test", streaming=True, trust_remote_code=True)
    ami = load_dataset("edinburghcstr/ami", "ihm", split="test", streaming=True, trust_remote_code=True)

    return {
        "LibriSpeech Clean": librispeech_clean,
        "LibriSpeech Other": librispeech_other,
        "Common Voice": common_voice,
        "VoxPopuli": voxpopuli,
        "TED-LIUM": tedlium,
        "Giga Speech": gigaspeech,
        "SPGI Speech": spgispeech,
        "AMI": ami
    }


def test_datasets(whisper_asr, wer_metric, eval_datasets, batch_size=16):
    wer_results = {}

    for dataset_name, dataset in eval_datasets.items():
        if DEBUG:
            dataset = dataset.take(DEBUG_SAMPLES)

        # resample to 16kHz
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

        # normalize references
        dataset = normalize_dataset(dataset, whisper_asr)

        # remove any empty references
        dataset = dataset.filter(is_target_text_in_range, input_columns=["norm_text"])

        predictions = []
        references = []
        for out in whisper_asr(data(dataset), batch_size=batch_size):
            if DEBUG:
                print(out)
            predictions.append(whisper_norm(whisper_asr, out["text"]))
            references.append(out["reference"][0])

        # compute the word error rate (WER)
        wer_result = wer_metric.compute(references=references, predictions=predictions)
        wer_result = round(100 * wer_result, 2)

        wer_results[dataset_name] = wer_result
        print(f"WER for data set {dataset_name}: {wer_result}")

    print(wer_results)

def main():
    #from huggingface_hub import login
    #login()

    wer_metric = evaluate.load("wer")
    eval_datasets = load_datasets()

    # Test PyTorch baseline
    #whisper_torch = load_torch_pipeline()
    #test_datasets(whisper_torch, wer_metric, eval_datasets)

    # Test Optimum Int4
    test_optimum_pipeline(eval_datasets)

if __name__ == "__main__":
    main()
