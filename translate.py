import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Path to the directory where you downloaded the files
model_path = "./nllb"  # Update this if your path is different

# Load tokenizer and model from local files
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

print("✅ Model and tokenizer loaded successfully.")


def translate(text, src_lang="eng_Latn", tgt_lang="hin_Deva"):
    """
    Translates text using NLLB-200.
    Default: English (eng_Latn) ➝ Hindi (hin_Deva)
    """
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Add source language token manually
    inputs["input_ids"] = tokenizer.convert_tokens_to_ids([src_lang]) + inputs["input_ids"].tolist()[0]
    inputs["input_ids"] = torch.tensor([inputs["input_ids"]])

    # Generate translation
    with torch.no_grad():
        output = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang)
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


# Example usage
english_text = "The patient should take one tablet after meals for four days."
hindi_translation = translate(english_text)
print("English:", english_text)
print("Hindi:", hindi_translation)

# User input translation
english_text2 = input("Enter English text: ")
hindi_translation2 = translate(english_text2)
print("English:", english_text2)
print("Hindi:", hindi_translation2)