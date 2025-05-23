#!/usr/bin/env python

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import re

import os
import pathlib
import api_keys
import sys
import gc

import numpy as np
# for generatring dispersion chart.

from torch import cuda, bfloat16
import torch
import transformers
import pandas as pd
import os
#import bitsandbytes as bnb_4bit_compute_dtype
import time
import itertools
from accelerate import Accelerator
torch.set_grad_enabled(False)

import numpy as np
# for generatring dispersion chart.

from torch import cuda, bfloat16
import torch
torch.cuda.empty_cache()


import warnings
warnings.filterwarnings('ignore')

import argparse
parser = argparse.ArgumentParser(description='Process text classification with various models in the dataset.')

from GPUtil import showUtilization as gpu_usage
from numba import cuda
def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()                             

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()

free_gpu_cache()

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

def cal_avg_prob(logits_dict):
    gen_word_logits = torch.stack([prob.squeeze(0) for prob in logits_dict])
    # to get log probabilities
    gen_word_probs = torch.nn.functional.softmax(gen_word_logits, dim=1)
    # avg over vocab distributions
    avg_probs = torch.mean(gen_word_probs, dim=0).detach().cpu().numpy()
    return avg_probs


def main():

    parser.add_argument('--dataset', help = 'give the name of the folder where our datasets are', required = True, type = str)
    parser.add_argument('--model', help = 'give the name of the model', type= str,
            default='google/gemma-7b-it')
    #parser.add_argument('--gpu_layers', help='if GPU layers present, add the number here', type = str, required=True, default = 0)
    parser.add_argument('--output_folder', help = 'give the name of the output folder', required = True, type = str)
    parser.add_argument('--output_file', help = 'give the name of the output file name', required = True, type = str)
    parser.add_argument('--token_num', help = 'give the number of maximum tokens to be generated', required = True, type = int, default = 10)
    parser.add_argument('--temperature', help = 'provide the temperature score', required = True, type = float, default = 0.5)
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()

    data = pd.read_csv(f'{args.dataset}.csv', encoding='utf-8')
    len_of_df = len(data)
    print(len_of_df)

    '''quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                #bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False,
            )'''

    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        #device_map="auto",
        #quantization_config=quant_config
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    output_list_1 = []
    vocab_prob_list_1 = []
    ppl_list_1 = []
    flag = False  # Defining the flag variable
    num = 0
    for row in data['text']:
        rate_limit_per_minute = 20
        delay = 60.0 / rate_limit_per_minute
        if num == len_of_df: # In case if the loop does not break
            flag = True
            print(f'Reached the {len_of_df}')
            break
        else:
            try:
                prompt = re.sub(r"\s+", " ", row)
                prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
                
                roleplay_prompt = f'''<bos><start_of_turn>user
                You are an expert in understanding slight linguistic nuances in the text, even when presented with texts that lack enough context. \
                You are well-versed with the political discourse/scenario in the United Kingdom since 2018, especially in social media platforms like Twitter. 

                Based on the past guidelines, please answer if the following text is any of the following 3 labels:
                - sexist
                - anti-sexist
                - neither
                The model should generate ONLY ONE of these labels as the output label, and nothing more or different. 
                
                Label the following text using the aforementioned instructions.
                Text: {prompt}
                Label:
                
                <end_of_turn>
                <start_of_turn>model
                '''

                messages = [
                    { "role": "user", "content": roleplay_prompt},
                ]

                text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                
                model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

                results = []
                avgs = []
                ppls = []
                for i in range(args.num_samples):

                    result_dict = model.generate(
                                    **model_inputs,
                                    do_sample=True,
                                    num_return_sequences=1,
                                    num_beams=1,
                                    max_new_tokens=args.token_num,
                                    temperature=args.temperature,
                                    #top_k=0,
                                    return_dict_in_generate=True,
                                    output_logits=True, # To get unprocessed prediction scores of the language modeling head
                                    output_hidden_states=False
                                )
                    decoded = tokenizer.batch_decode(result_dict["sequences"])
                    result = decoded[0].split("<end_of_turn>")[-1].replace("<eos>", "")
                    result = (result.split(f"Label:")[-1]).split(f"\n\n")[0].strip().lower()

                    avg_probs = cal_avg_prob(result_dict['logits'])

                    transition_scores = model.compute_transition_scores(
                            result_dict.sequences, result_dict.logits, normalize_logits=True
                        )
                        
                    ppl = np.exp(-np.mean(transition_scores[0].float().cpu().numpy()))

                    results.append(result)
                    avgs.append(avg_probs)
                    ppls.append(ppl)

                    del result, avg_probs, ppl
                
                output_list_1.append(results)
                vocab_prob_list_1.append(avgs)
                ppl_list_1.append(ppls)
                print(f'Finished iteration {num} for 1st one')
                num += 1
                
                del results, avgs, ppls
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(delay)

            except Exception as e:
                #print(f'Error type {e.message}')
                print(f'Error type {e}')
                print(f'Since it is taking longer for {num} in 1st one, we use the default label here')
                num += 1
                output_list_1.append('neither')
                vocab_prob_list_1.append([])
                ppl_list_1.append(1)
                torch.cuda.empty_cache()
                gc.collect()
                print('Will come back in 2 minutes ...')
                time.sleep(2 * 60) # 2 minutes
        if flag == True:
            print('Reached the whole length of the dataset. Stopping now')
            break

    data['roleplay_results'] = output_list_1
    data['roleplay_vocab_probs'] = vocab_prob_list_1
    data['roleplay_ppl'] = ppl_list_1

    data.to_csv(f'{args.output_folder}/{args.output_file}.csv')

    del output_list_1, vocab_prob_list_1, ppl_list_1
    time.sleep(5 * 60)
    torch.cuda.empty_cache()
    gc.collect()

    output_list_2 = []
    vocab_prob_list_2 = []
    ppl_list_2 = []

    flag = False  # Defining the flag variable
    num = 0
    for row in data['text']:
        rate_limit_per_minute = 20
        delay = 60.0 / rate_limit_per_minute
        if num == len_of_df: # In case if the loop does not break
            flag = True
            print(f'Reached the {len_of_df}')
            break
        else:
            try:
                prompt = re.sub(r"\s+", " ", row)
                prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()

                content_prompt = f'''<bos><start_of_turn>user
                You are an expert in understanding slight linguistic nuances in the text, even when presented with texts that lack enough context. \
                You are well-versed with the political discourse/scenario in the United Kingdom since 2018, especially in social media platforms like Twitter. 

                ### Read on for detailed explanations. Something can be sexist because of its `content` (on the political or non-political incident in question).

                ### Instructions for understanding `content`:

                ## Sexist: Texts formulating a prescriptive set of behaviors or qualities, \
                that women (and men) are supposed to exhibit in order to conform to traditional gender roles. This could be texts formulating \
                a descriptive set of properties  that  supposedly  differentiates the two genders and expressed through explicit or implicit comparisons and perpetuating gender-based stereotypes. \
                Aside from acknowledging the inequalities, these texts could be endorsing or justifying them in a non-flattering manner. This may contain texts \
                stating that there are no inequalities between men and women (any more) and/or that are opposing feminism. They might possess views which indicate \
                women are not competent adults, or women having favourable traits that men stereotypically lack. Also, it could mean \
                viewing women as romantic objects with a genuine desire for psychological closeness. Texts could contain comments or actions that cause or \
                aggravate restrictions on how women communicate. Texts could also have promotion of discriminatory codes of conduct for women in the guise of morality; \
                also applies to statements that feed into such codes and narratives. 'Sexist' attitudes can also be indirect attempting to deny responsibility for an utterance, mediating through irony or by disguising \
                the force of sexism of utterance through humour and innuendo, embedding sexism sexism at the level of presupposition, or prefacing sexist statements with disclaimers or hesitation. \
                The text can contain statements that question the rights and treatment of women in a manner that reflects gender bias or sexism. \
                Sexism could also be institutional, where the sexist attitude drawn upon repeatedly by various institutions become normalised. 

                ## Anti-sexist: It is the belief in and advocacy for the equality of the sexes and the elimination of sexism and gender-based discrimination or biasness. \
                It can indicate the user's interest to support progress towards non-violence of women and other genders, and also promoting gender equality. \
                In other words, anti-sexism demonstrates increasing prevelance of active behaviors challenging gender-based discrimination , i.e. anti-prejudicial sexist actions, aside from supporting reduction of the said gender-based prejudiced attitudes. \
                It is a commitment to challenging and combating the various forms of prejudice, bias, and inequality that can be directed toward individuals based on their gender or sex. \
                Texts could include reprimanding sexism both at a structural and institutional level; actively opposing the spread and occurence of events/discussions leading to sexism through challenging and confronting sexist behaviors; or attitudes of someone through a direct or a quoted text. \  
                This can involve promoting gender equality, advocating for women's rights, challenging harmful gender stereotypes, showing willingness to intervene in preventing gender-based violence or discrimination, and condemning sexist actions. \
                It also extends to supporting the rights and equality of individuals of all gender identities and expressions. Such texts might recognize male privileges. \
                This could be texts having strong language and sarcastic tone, but overall meaning and intention of the text should indicate 'anti-sexist' rhetoric.  \
                Such texts could contain strong words in favor of demolishing institutional and structural practices of sexism and gender inequality; and may attack or rebuke targeted texts (e.g. quotes) or individuals for their sexist remarks. \
                Willingness to intervene to prevent different forms of 'sexism' - be it hostile (e.g., violence and/or sexual assault against women) or benevolent (subtle and implicit gendered discrimination) or indirect, is an important behavioural component of 'anti-sexist' attitude.
                

                Based on the past guidelines, please answer if the following text is any of the following 3 labels:
                1) sexist
                2) anti-sexist
                3) neither
                The model should generate ONLY ONE of these labels as the output label, and nothing more or different. 

                Label the following text using the aforementioned instructions.
                Text: {prompt}
                Label:
                <end_of_turn>
                <start_of_turn>model
                '''
                messages = [
                    { "role": "user", "content": content_prompt},
                ]

                text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                
                model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

                results = []
                avgs = []
                ppls = []
                for i in range(args.num_samples):

                    result_dict = model.generate(
                                    **model_inputs,
                                    do_sample=True,
                                    num_return_sequences=1,
                                    num_beams=1,
                                    max_new_tokens=args.token_num,
                                    temperature=args.temperature,
                                    #top_k=0,
                                    return_dict_in_generate=True,
                                    output_logits=True, # To get unprocessed prediction scores of the language modeling head
                                    output_hidden_states=False
                                )
                    decoded = tokenizer.batch_decode(result_dict["sequences"])
                    result = decoded[0].split("<end_of_turn>")[-1].replace("<eos>", "")
                    result = (result.split(f"Label:")[-1]).split(f"\n\n")[0].strip().lower()

                    avg_probs = cal_avg_prob(result_dict['logits'])

                    transition_scores = model.compute_transition_scores(
                            result_dict.sequences, result_dict.logits, normalize_logits=True
                        )
                        
                    ppl = np.exp(-np.mean(transition_scores[0].float().cpu().numpy()))

                    results.append(result)
                    avgs.append(avg_probs)
                    ppls.append(ppl)

                    del result, avg_probs, ppl
                
                print(results)
                output_list_2.append(results)
                vocab_prob_list_2.append(avgs)
                ppl_list_2.append(ppls)
                print(f'Finished iteration {num} for 2nd one')
                num += 1
                
                del results, avgs, ppls
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(delay)

            except Exception as e:
                #print(f'Error type {e.message}')
                print(f'Error type {e}')
                print(f'Since it is taking longer for {num} in 2nd one, we use the default label here')
                num += 1
                output_list_2.append('neither')
                vocab_prob_list_2.append([])
                ppl_list_2.append(1)
                torch.cuda.empty_cache()
                gc.collect()
                print('Will come back in 2 minutes ...')
                time.sleep(2 * 60) # 2 minutes
        if flag == True:
            print('Reached the whole length of the dataset. Stopping now')
            break

    data['content_results'] = output_list_2
    data['content_vocab_probs'] = vocab_prob_list_2
    data['content_ppl'] = ppl_list_2

    data.to_csv(f'{args.output_folder}/{args.output_file}.csv')

    del output_list_2, vocab_prob_list_2, ppl_list_2
    time.sleep(5 * 60)
    torch.cuda.empty_cache()
    gc.collect()


    output_list_3 = []
    vocab_prob_list_3 = []
    ppl_list_3 = []

    flag = False  # Defining the flag variable
    num = 0
    for row in data['text']:
        rate_limit_per_minute = 20
        delay = 60.0 / rate_limit_per_minute
        if num == len_of_df: # In case if the loop does not break
            flag = True
            print(f'Reached the {len_of_df}')
            break
        else:
            try:
                prompt = re.sub(r"\s+", " ", row)
                prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()

                zeroshot_prompt = f'''<bos><start_of_turn>user
                You are an expert in understanding slight linguistic nuances in the text, even when presented with texts that lack enough context. \
                You are well-versed with the political discourse/scenario in the United Kingdom since 2018, especially in social media platforms like Twitter. 
                
                ### Read on for detailed explanations. Something can be 'sexist' because of its `context` (on the political or non-political incident in question), \
                because of its `content` (what the speaker believes), and because of its `phrasing` (because of the speaker’s choice of words).

                ### The `context` is mostly political and it consists of texts that revolve around discussions of policies, government actions, ideologies, elections, etc. \
                These texts would aim to engage with societal issues, power dynamics, and decision-making processes within the realm of public affairs. \
                Pertaining to the practice and theory of influencing other people on a civic or individual level, often concerning government or public affairs. \
                Reference to any of the target's current or former political and/or behavioral activity. This could be an implicit indication \
                in the text, or a direct implication through mentions of their position on a certain topic. A typical political text could have strong language, \
                a harsh tone and slurs; and question the political standing and political opinion of the target (usually indicated by mention of policies or government strategies) \
                or the political position the target holds. Yet it should not undermine the intelligence of the target.

                ## Sexist due to `content`. A text may be sexist if the speaker shows a prescriptive set of behaviors or qualities, \
                that women (and men) are supposed to exhibit in order to conform to traditional gender roles. This could be texts formulating \
                a descriptive set of properties  that  supposedly  differentiates the two genders and expressed through explicit or implicit comparisons and perpetuating gender-based stereotypes. \
                Aside from acknowledging the inequalities, these texts could be endorsing or justifying them in a non-flattering manner. 

                ## Anti-sexist due to `content`: A text may be anti-sexist if the speaker believe in and advocacy for the equality of the sexes and the elimination of sexism and gender-based discrimination or biasness. \
                It can indicate the user's interest to support progress towards non-violence of women and other genders, and also promoting gender equality. \
                In other words, anti-sexism demonstrates increasing prevelance of active behaviors challenging gender-based discrimination , i.e. anti-prejudicial sexist actions, aside from supporting reduction of the said gender-based prejudiced attitudes. \
                It is a commitment to challenging and combating the various forms of prejudice, bias, and inequality that can be directed toward individuals based on their gender or sex. 
                
                ## Sexist due to `phrasing`: Texts may be sexist simply because of how the speaker phrases it–independently from what general beliefs or attitudes the speaker holds. 
                A message is sexist, for example, when it contains attacks, foul language, or derogatory depictions directed towards individuals because of their gender.

                ## Anti-sexist due to `phrasing`: Texts could include reprimanding sexism both at a structural and institutional level; actively opposing the spread and occurence of events/discussions \
                leading to sexism through challenging and confronting sexist behaviors; or attitudes of someone through a direct or a quoted text. \
                Texts could be offensive and have a harsh tone, explicitly calling out a target's sexist position or their sexist opinion on any issue, but not because of the target's sex or gender. 

                ## Sexist due to `context`: Texts could be mocking female perspectives from female politicians, minimize their political contributions and undermine their achievements. \
                It can also question their commitments to public office by implicating that they should focus more on their family commitments, and their political performance being compared to their capability in \
                familial setting. They may also publish appearance-centric criticism of the female politicians, unlike their male counterparts. They tone could be ironic and exaggerated, and often in the guise of humour.

                ## Anti-sexist due to `context`: Texts could promote female perspectives from female politicians, rebuke political gender differences, and also uplifting and encouraging more female political participations.

                Based on the past guidelines, please answer if the following text is any of the following 3 labels:
                1) sexist
                2) anti-sexist
                3) neither
                The model should generate ONLY ONE of these labels as the output label, and nothing more or different. 

                Label the following text using the aforementioned instructions.
                Text: {prompt}
                Label:
                <end_of_turn>
                <start_of_turn>model
                '''
                messages = [
                    { "role": "user", "content": zeroshot_prompt},
                ]

                text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                
                model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

                results = []
                avgs = []
                ppls = []
                for i in range(args.num_samples):

                    result_dict = model.generate(
                                    **model_inputs,
                                    do_sample=True,
                                    num_return_sequences=1,
                                    num_beams=1,
                                    max_new_tokens=args.token_num,
                                    temperature=args.temperature,
                                    #top_k=0,
                                    return_dict_in_generate=True,
                                    output_logits=True, # To get unprocessed prediction scores of the language modeling head
                                    output_hidden_states=False
                                )
                    decoded = tokenizer.batch_decode(result_dict["sequences"])
                    result = decoded[0].split("<end_of_turn>")[-1].replace("<eos>", "")
                    result = (result.split(f"Label:")[-1]).split(f"\n\n")[0].strip().lower()

                    avg_probs = cal_avg_prob(result_dict['logits'])

                    transition_scores = model.compute_transition_scores(
                            result_dict.sequences, result_dict.logits, normalize_logits=True
                        )
                        
                    ppl = np.exp(-np.mean(transition_scores[0].float().cpu().numpy()))

                    results.append(result)
                    avgs.append(avg_probs)
                    ppls.append(ppl)

                    del result, avg_probs, ppl
                
                output_list_3.append(results)
                vocab_prob_list_3.append(avgs)
                ppl_list_3.append(ppls)
                print(f'Finished iteration {num} for 3rd one')
                num += 1
                
                del results, avgs, ppls
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(delay)

            except Exception as e:
                #print(f'Error type {e.message}')
                print(f'Error type {e}')
                print(f'Since it is taking longer for {num} in 3rd one, we use the default label here')
                num += 1
                output_list_3.append('neither')
                vocab_prob_list_3.append([])
                ppl_list_3.append(1)
                torch.cuda.empty_cache()
                gc.collect()
                print('Will come back in 2 minutes ...')
                time.sleep(2 * 60) # 2 minutes
        if flag == True:
            print('Reached the whole length of the dataset. Stopping now')
            break

    data['zeroshot_results'] = output_list_3
    data['zeroshot_vocab_probs'] = vocab_prob_list_3
    data['zeroshot_ppl'] = ppl_list_3

    data.to_csv(f'{args.output_folder}/{args.output_file}.csv')

    del output_list_3, vocab_prob_list_3, ppl_list_3
    time.sleep(5 * 60)
    torch.cuda.empty_cache()
    gc.collect()



    output_list_4 = []
    vocab_prob_list_4 = []
    ppl_list_4 = []

    flag = False  # Defining the flag variable
    num = 0
    for row in data['text']:
        rate_limit_per_minute = 20
        delay = 60.0 / rate_limit_per_minute
        if num == len_of_df: # In case if the loop does not break
            flag = True
            print(f'Reached the {len_of_df}')
            break
        else:
            try:
                prompt = re.sub(r"\s+", " ", row)
                prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()

                fewshot_prompt = f'''<bos><start_of_turn>user
                You are an expert in understanding slight linguistic nuances in the text, even when presented with texts that lack enough context. \
                You are well-versed with the political discourse/scenario in the United Kingdom since 2018, especially in social media platforms like Twitter. 
                
                ### Read on for detailed explanations. Something can be 'sexist' because of its `context` (on the political or non-political incident in question), \
                because of its `content` (what the speaker believes), and because of its `phrasing` (because of the speaker’s choice of words).

                ### The `context` is mostly political and it consists of texts that revolve around discussions of policies, government actions, ideologies, elections, etc. \
                These texts would aim to engage with societal issues, power dynamics, and decision-making processes within the realm of public affairs. \
                Pertaining to the practice and theory of influencing other people on a civic or individual level, often concerning government or public affairs. \
                Reference to any of the target's current or former political and/or behavioral activity. This could be an implicit indication \
                in the text, or a direct implication through mentions of their position on a certain topic. A typical political text could have strong language, \
                a harsh tone and slurs; and question the political standing and political opinion of the target (usually indicated by mention of policies or government strategies) \
                or the political position the target holds. Yet it should not undermine the intelligence of the target.

                ## Sexist due to `content`. A text may be sexist if the speaker shows a prescriptive set of behaviors or qualities, \
                that women (and men) are supposed to exhibit in order to conform to traditional gender roles. This could be texts formulating \
                a descriptive set of properties  that  supposedly  differentiates the two genders and expressed through explicit or implicit comparisons and perpetuating gender-based stereotypes. \
                Aside from acknowledging the inequalities, these texts could be endorsing or justifying them in a non-flattering manner. 

                ## Anti-sexist due to `content`: A text may be anti-sexist if the speaker believe in and advocacy for the equality of the sexes and the elimination of sexism and gender-based discrimination or biasness. \
                It can indicate the user's interest to support progress towards non-violence of women and other genders, and also promoting gender equality. \
                In other words, anti-sexism demonstrates increasing prevelance of active behaviors challenging gender-based discrimination , i.e. anti-prejudicial sexist actions, aside from supporting reduction of the said gender-based prejudiced attitudes. \
                It is a commitment to challenging and combating the various forms of prejudice, bias, and inequality that can be directed toward individuals based on their gender or sex. 
                
                ## Sexist due to `phrasing`: Texts may be sexist simply because of how the speaker phrases it–independently from what general beliefs or attitudes the speaker holds. 
                A message is sexist, for example, when it contains attacks, foul language, or derogatory depictions directed towards individuals because of their gender.

                ## Anti-sexist due to `phrasing`: Texts could include reprimanding sexism both at a structural and institutional level; actively opposing the spread and occurence of events/discussions \
                leading to sexism through challenging and confronting sexist behaviors; or attitudes of someone through a direct or a quoted text. \
                Texts could be offensive and have a harsh tone, explicitly calling out a target's sexist position or their sexist opinion on any issue, but not because of the target's sex or gender. 

                ## Sexist due to `context`: Texts could be mocking female perspectives from female politicians, minimize their political contributions and undermine their achievements. \
                It can also question their commitments to public office by implicating that they should focus more on their family commitments, and their political performance being compared to their capability in \
                familial setting. They may also publish appearance-centric criticism of the female politicians, unlike their male counterparts. They tone could be ironic and exaggerated, and often in the guise of humour.

                ## Anti-sexist due to `context`: Texts could promote female perspectives from female politicians, rebuke political gender differences, and also uplifting and encouraging more female political participations.

                Based on the past guidelines, please answer if the following text is any of the following 3 labels:
                - sexist
                - anti-sexist
                - neither

                The model should generate ONLY ONE of these labels as the output label, and nothing more or different. 

                Here are a few examples to demonstrate how to label texts accordingly:

                1.  Text: 'Yes, but she was asking for it!'? When will people grow up and stop sexualizing women?!
                    Label: anti-sexist
                2.  Text: The conference will cover various topics, including technology and art. Gender biases will not be tolerated.
                    Label: anti-sexist
                3.  Text: Give her a pity party because of some news article. She has been vile attacking people for many years if they do not agree with her politics calling people scum.
                    Label: neither
                4.  Text: Was she flashing her minge?
                    Label: sexist
                5.  Text: Look the way she dressed! She's definitely asking for it.🙄
                    Label: sexist
                6. Text: Men will be men. It is true, and you should accept it.
                    Label: sexist
                7. Text: Female politicians often face biased media coverage compared to their male counterparts.
                    Label: anti-sexist

                Label the following text using the aforementioned instructions.
                Text: {prompt}
                Label:
                <end_of_turn>
                <start_of_turn>model
                '''
                messages = [
                    { "role": "user", "content": fewshot_prompt},
                ]

                text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                
                model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

                results = []
                avgs = []
                ppls = []
                for i in range(args.num_samples):

                    result_dict = model.generate(
                                    **model_inputs,
                                    do_sample=True,
                                    num_return_sequences=1,
                                    num_beams=1,
                                    max_new_tokens=args.token_num,
                                    temperature=args.temperature,
                                    #top_k=0,
                                    return_dict_in_generate=True,
                                    output_logits=True, # To get unprocessed prediction scores of the language modeling head
                                    output_hidden_states=False
                                )
                    decoded = tokenizer.batch_decode(result_dict["sequences"])
                    result = decoded[0].split("<end_of_turn>")[-1].replace("<eos>", "")
                    result = (result.split(f"Label:")[-1]).split(f"\n\n")[0].strip().lower()

                    avg_probs = cal_avg_prob(result_dict['logits'])

                    transition_scores = model.compute_transition_scores(
                            result_dict.sequences, result_dict.logits, normalize_logits=True
                        )
                        
                    ppl = np.exp(-np.mean(transition_scores[0].float().cpu().numpy()))

                    results.append(result)
                    avgs.append(avg_probs)
                    ppls.append(ppl)

                    del result, avg_probs, ppl
                
                output_list_4.append(results)
                vocab_prob_list_4.append(avgs)
                ppl_list_4.append(ppls)
                print(f'Finished iteration {num} for 4th one')
                num += 1
                
                del results, avgs, ppls
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(delay)

            except Exception as e:
                #print(f'Error type {e.message}')
                print(f'Error type {e}')
                print(f'Since it is taking longer for {num} in 4th one, we use the default label here')
                num += 1
                output_list_4.append('neither')
                vocab_prob_list_4.append([])
                ppl_list_4.append(1)
                torch.cuda.empty_cache()
                gc.collect()
                print('Will come back in 2 minutes ...')
                time.sleep(2 * 60) # 2 minutes
        if flag == True:
            print('Reached the whole length of the dataset. Stopping now')
            break

    data['fewshot_results'] = output_list_4
    data['fewshot_vocab_probs'] = vocab_prob_list_4
    data['fewshot_ppl'] = ppl_list_4

    data.to_csv(f'{args.output_folder}/{args.output_file}.csv')

    del output_list_4, vocab_prob_list_4, ppl_list_4
    time.sleep(5 * 60)
    torch.cuda.empty_cache()
    gc.collect()

    del args


if __name__ == '__main__':
    main()

