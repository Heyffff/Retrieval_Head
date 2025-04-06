import sys
sys.path.append("./faiss_attn/")
import torch
import argparse
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from source.modeling_llama import LlamaForCausalLM
from source.modeling_qwen2 import Qwen2ForCausalLM
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def decode(model, tokenizer, q_outputs, inp, max_decode_len):
    generated_ids = []
    decode_attention = []
    layer_idx, head_idx = 19, 5  # 检索头编号
    past_kv = q_outputs.past_key_values
    for step_i in range(max_decode_len):
        inp = inp.view(1, 1)
        outputs = model(input_ids=inp, past_key_values=past_kv, use_cache=True, output_attentions=True, attn_mode="torch" )
        past_kv = outputs.past_key_values
        inp = outputs.logits[0, -1].argmax()
        step_token = tokenizer.convert_ids_to_tokens(inp.item())
        if step_token in ['<|endoftext|>', '<|end_of_text|>', '<|eot_id|>']: break
        generated_ids.append(inp.item())
        decode_attention.append(outputs.attentions[layer_idx][0][head_idx][-1])
    return generated_ids, decode_attention


def plot_attention_scores(attention_matrix, input_tokens, generated_words, model):
    top_k = 3
    new_attention_matrix = np.zeros(attention_matrix.shape)
    for row in range(attention_matrix.shape[0]):
        sorted_indices = np.argsort(attention_matrix[row, :])[::-1][:top_k]
        for rank, idx in enumerate(sorted_indices):
            new_attention_matrix[row, idx] = top_k - rank
    plt.figure(figsize=(80, 64))
    plt.imshow(new_attention_matrix, cmap='Blues')
    plt.xticks(np.arange(len(input_tokens)), input_tokens, rotation=45, ha='left')
    plt.yticks(np.arange(len(generated_words)), generated_words)
    plt.gca().xaxis.tick_top()
    plt.gca().yaxis.tick_right()
    plt.tight_layout()
    plt.savefig(f"./viz/attention_score/{model}.png")


def find_special_token_ranges(token_list):
    ranges = []
    start_index = 0
    for i, token in enumerate(token_list):
        has_non_chinese = any(not (('\u4e00' <= char <= '\u9fff') or ('0' <= char <= '9')) for char in token)
        if has_non_chinese:
            if i > start_index:
                ranges.append([start_index, i])
            start_index = i + 1
    if start_index < len(token_list):
        ranges.append([start_index, len(token_list)])
    return ranges


if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--sentence', action="store_true")
    parser.add_argument('--only_chunk', action="store_true")
    args = parser.parse_args()
    max_decode_len = 150

    # 设置字体
    plt.rcParams['font.family'] = 'AR PL UMing CN'
    plt.rcParams.update({'font.size': 72 if args.sentence else 16, 'font.weight': 'bold'})

    # prompt
    # chunks = "（一）规范完善低保准入条件。落实最低生活保障审核确认相关法规文件对低保条件的有关规定，在综合考虑申请家庭收入、财产状况等的基础上，做好低保审核确认工作。不得随意附加非必要限制性条件，不得以特定职业、特殊身份等为由，或者未经家庭经济状况调查核实直接认定申请家庭符合或者不符合条件。申请家庭符合条件的，不得仅将个别家庭成员纳入低保范围。采取“劳动力系数”等方式核算申请家庭收入的，要客观考虑家庭成员实际情况，对确实难以就业或者较长时间无法获得收入的，根据家庭实际困难情况综合判断是否纳入低保范围。成年无业重度残疾人可以参照“单人户”提出低保申请。依靠兄弟姐妹或者60周岁及以上老年人供养的成年无业重度残疾人，在评估认定其家庭经济状况时，兄弟姐妹或者60周岁及以上老年人给付的供养费用，可以视情适当豁免，符合条件的，纳入低保范围。"
    # chunks = "当事人无法出具居民户口簿的，婚姻登记机关可凭公安部门或有关户籍管理机构出具的加盖印章的户籍证明办理婚姻登记；当事人属于集体户口的，婚姻登记机关可凭集体户口簿内本人的户口卡片或加盖单位印章的记载其户籍情况的户口簿复印件办理婚姻登记。当事人未办理落户手续的，户口迁出地或另一方当事人户口所在地的婚姻登记机关可凭公安部门或有关户籍管理机构出具的证明材料办理婚姻登记。"
    # chunks = "第五条 符合下列情形之一的，应当认定为本办法所称的无劳动能力： （一）60 周岁以上的老年人； （二）未满 16 周岁的未成年人； （四）省、自治区、直辖市人民政府规定的其他情形。 第六条 收入低于当地最低生活保障标准，且财产符合当地特困人员财产状况规定的，应当认定为本办法所称的无生活来源。 前款所称收入包括工资性收入、经营净收入、财产净收入、转移净收入等各类收入。中央确定的城乡居民基本养老保险基础养老金、基本医疗保险等社会保险和优待抚恤金、高龄津贴不计入在内。\n其中，转移性收入指国家、机关企事业单位、社会组织对居民的各种经常性转移支付和居民之间的经常性收入转移，包括赡养（抚养、扶养）费、离退休金、失业保险金、遗属补助金、赔偿收的经常性转移支出，包括缴纳的税款、各项社会保障支出、赡养支出以及其他经常性转移支出等。 （五）其他应当计入家庭收入的项目。 下列收入不计入家庭收入： （一）国家规定的优待抚恤金、计划生育奖励与扶助金、奖学金、见义勇为等奖励性补助； （二）政府发放的各类社会救助款物； （三）“十四五”期间，中央确定的城乡居民基本养老保险基础养老金； （四）设区的市级以上地方人民政府规定的其他收入。"
    chunks = "试点地区内地居民结(离)婚登记扩大至一方当事人常住户口所在地或经常居住地婚姻登记机关办理。双方均非本地户籍的当事人可凭一方居住证和双方户口簿、身份证，在居住证发放地或一方常住户口所在地婚姻登记机关申请办理婚姻登记。\n\n三、试点地区和试点期限\n（一）试点地区。辽宁省、山东省、广东省、重庆市、四川省实施内地居民结婚登记和离婚登记“跨省通办”试点，江苏省、河南省、湖北省武汉市、陕西省西安市实施内地居民结婚登记“跨省通办”试点。\n（二）试点期限。试点期限为2年，自2021年6月1日起至2023年5月31日止。"
    # question = "依靠兄弟姐妹或者60周岁及以上老年人供养的成年无业重度残疾人应该怎么评定家庭收入？"
    # question = "我想离婚，但找不到户口本了，该怎么办？"
    # question = "在申请低保时，发给牺牲的军人或者烈士家属的遗属补助金计算收入吗？"
    question = "男方户口湖北，女方户口广西，两人都在西安工作，可以在西安领证吗？"
    prompt = "{chunks}\n请你根据上述提供的文档内容回答问题：{question}\n以下是你的回答:\n"

    # 读取模型
    model_path = f"../model/{args.model}"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if "qwen" in args.model.lower():
        model = Qwen2ForCausalLM.from_pretrained(
            model_path, torch_dtype="auto", device_map='auto', use_flash_attention_2="flash_attention_2"
        ).eval()
    else:
        model = LlamaForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map='auto', use_flash_attention_2="flash_attention_2"
        ).eval()

    # decoding
    input_ids = tokenizer(prompt.replace("{chunks}", chunks).replace("{question}", question), return_tensors="pt")['input_ids']
    q_outputs = model(input_ids=input_ids[:, :-1], use_cache=True, return_dict=True)
    output_ids, decode_attention = decode(model, tokenizer, q_outputs, input_ids[0][-1], max_decode_len)
    decode_attention = pad_sequence(decode_attention, batch_first=True)
    if args.only_chunk:  # 如果只观察decode对于chunk的注意力分数
        chunk_ids = tokenizer(chunks, return_tensors="pt")['input_ids']
        length = len(chunk_ids[0])
        input_id_list = chunk_ids[0].tolist()
    else:  # 反之除chunk外还会加上question
        length = len(input_ids[0])
        input_id_list = input_ids[0].tolist()
    decode_attention = decode_attention[:, :length]

    # 基于id解码出input_tokens和output_tokens
    input_tokens, output_tokens = [], []
    for i in input_id_list:
        input_tokens.append(tokenizer.decode(i))
    for i in output_ids:
        output_tokens.append(tokenizer.decode(i))

    # 合并tokens及attention至sentence级别
    input_ranges = find_special_token_ranges(input_tokens)
    output_ranges = find_special_token_ranges(output_tokens)
    input_sentences, output_sentences = [], []
    sentence_attention = torch.empty(0)
    for r in input_ranges:
        input_sentences.append(''.join(input_tokens[r[0]:r[1]]))
        sentence_attention = torch.cat((sentence_attention, torch.mean(decode_attention[:, r[0]:r[1]], dim=1, keepdim=True)), dim=1)
    sentence_attention_final = torch.empty(0)
    for r in output_ranges:
        output_sentences.append(''.join(output_tokens[r[0]:r[1]]))
        sentence_attention_final = torch.cat((sentence_attention_final, torch.mean(sentence_attention[r[0]:r[1]], dim=0, keepdim=True)), dim=0)

    # 绘制热力图
    if args.sentence:
        plot_attention_scores(sentence_attention_final.detach().numpy(), input_sentences, output_sentences, args.model)
    else:
        plot_attention_scores(decode_attention.detach().to(torch.float32).numpy(), input_tokens, output_tokens, args.model)
