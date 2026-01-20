import os
import argparse
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams

# 假设原有的 utils, models, judge, code_utils 都在路径中
from utils import load_data, write_json, read_json, extract_code
from judge import Judger
from code_utils import code_executor, estimate_pass_at_k

# ==========================================
# 1. vLLM 模型封装
# ==========================================
class VLLMWrapper:
    def __init__(self, model_path, tp_size=8, gpu_memory=0.9):
        print(f"Loading vLLM model from {model_path} with TP={tp_size}...")
        self.llm = LLM(
            model=model_path, 
            tensor_parallel_size=tp_size, 
            gpu_memory_utilization=gpu_memory,
            trust_remote_code=True
        )

    def generate(self, prompts, n=1, temperature=0.0):
        # n > 1 用于 CS 题目的 pass@k
        sampling_params = SamplingParams(
            n=n,
            temperature=temperature,
            max_tokens=8192,
            stop=["<|endoftext|>", "### Instruction", "Questions:"] # 根据模型调整停止词
        )
        outputs = self.llm.generate(prompts, sampling_params)
        
        # 返回结果列表。如果是 n=1, 每个元素是单条字符串；如果是 n=5, 每个元素是 5 条字符串的列表
        res = []
        for output in outputs:
            if n == 1:
                res.append(output.outputs[0].text)
            else:
                res.append([o.text for o in output.outputs])
        return res

# ==========================================
# 2. 推理逻辑 (Inference)
# ==========================================
def run_inference(args, datasets, vllm_model):
    non_cs_tasks = [ex for ex in datasets if ex["subject"] != "CS"]
    cs_tasks = [ex for ex in datasets if ex["subject"] == "CS"]

    judger = Judger()
    
    # --- 处理非 CS 题目 ---
    if non_cs_tasks:
        print(f"Running inference for {len(non_cs_tasks)} non-CS tasks...")
        # 提取 Prompt (这里假设 example["prompt"] 是构造好的字符串，如果不是，需在此构造)
        prompts = [ex.get("prompt", str(ex)) for ex in non_cs_tasks] 
        results = vllm_model.generate(prompts, n=1)

        for example, output_text in zip(non_cs_tasks, results):
            save_path = os.path.join(args.model_output_dir, example["subject"], f"{example['id']}.json")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            model_answer = judger.extract_boxed_answer(output_text)
            write_json({
                "id": example["id"],
                "answer_type": example["answer_type"],
                "type_sequence": example["type_sequence"],
                "model_output": output_text,
                "model_answer": model_answer
            }, save_path)

    # --- 处理 CS 题目 (pass@5) ---
    if cs_tasks:
        print(f"Running inference for {len(cs_tasks)} CS tasks (n=5)...")
        prompts = [ex.get("prompt", str(ex)) for ex in cs_tasks]
        results = vllm_model.generate(prompts, n=5, temperature=0.2)

        for example, output_list in zip(cs_tasks, results):
            save_path = os.path.join(args.model_output_dir, example["subject"], f"{example['id']}.json")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 提取代码
            code_snippets = [code for text in output_list if (code := extract_code(text))]
            write_json({
                "id": example["id"],
                "answer_type": example["answer_type"],
                "type_sequence": example["type_sequence"],
                "model_output": output_list,
                "model_answer": code_snippets
            }, save_path)

# ==========================================
# 3. 评估逻辑 (Evaluation)
# ==========================================
def run_evaluation(args, datasets):
    # 先聚合所有推理产生的结果
    output_dict = {}
    for root, dirs, files in os.walk(args.model_output_dir):
        for file in files:
            if file.endswith(".json"):
                d = read_json(os.path.join(root, file))
                output_dict[d["id"]] = d.get("model_answer", None)

    judger = Judger()
    results_records = {}

    print("Evaluating results...")
    for example in tqdm(datasets):
        problem_id = example["id"]
        res_dir = os.path.join(args.result_dir, example["subject"])
        os.makedirs(res_dir, exist_ok=True)
        
        model_ans = output_dict.get(problem_id)
        result = 0

        if example["subject"] != "CS":
            answer = example["answer"][0] if isinstance(example["answer"], list) and len(example["answer"])==1 else example["answer"]
            if model_ans is not None:
                result = judger.auto_judge(model_ans, answer, example["type_sequence"])
            else:
                result = False
        else:
            # CS 评估: Code Executor
            if model_ans:
                code_results = code_executor(model_ans, example["test_cases"])
                num_correct = sum(c == "Passed" for c in code_results)
                result = estimate_pass_at_k([len(model_ans)], [num_correct], 1)[0]
            else:
                result = 0

        # 保存单个结果
        record = {
            "id": problem_id, "answer_type": example["answer_type"], "result": result,
            "subject": example["subject"], "language": example["language"], "modality": example["modality"]
        }
        write_json(record, os.path.join(res_dir, f"{problem_id}.json"))
        results_records[problem_id] = record

    return results_records

# ==========================================
# 4. 统计与主函数
# ==========================================
def calculate_and_print(results_dict):
    # 复用原来的统计逻辑
    from evaluation import calculate_statistics, print_statistics
    total_rate, subject_rate, language_rate, modality_rate = calculate_statistics(results_dict)
    print_statistics(total_rate, subject_rate, language_rate, modality_rate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 路径参数
    parser.add_argument('--hf_data_path', type=str, default="GAIR/OlympicArena")
    parser.add_argument('--model_output_dir', type=str, default="./model_output/")
    parser.add_argument('--result_dir', type=str, default="./result/")
    parser.add_argument("--split", type=str, default="val")
    
    # vLLM 参数
    parser.add_argument("--model_path", type=str, required=True, help="Local path or HF ID")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gpu_mem", type=float, default=0.9)
    
    args = parser.parse_args()
    
    # 初始化路径
    model_name = os.path.basename(args.model_path.rstrip('/'))
    args.model_output_dir = os.path.join(args.model_output_dir, model_name, args.split)
    args.result_dir = os.path.join(args.result_dir, model_name, args.split)
    
    # 1. 加载数据
    datasets = load_data(args.hf_data_path, args.split)
    
    # 2. 推理 (Inference)
    vllm_model = VLLMWrapper(args.model_path, args.tp, args.gpu_mem)
    run_inference(args, datasets, vllm_model)
    
    # 释放显存给评估（如果评估需要跑代码或Reward Model的话，虽然这里主要跑CPU代码执行）
    del vllm_model
    
    # 3. 评估 (Evaluation)
    results_dict = run_evaluation(args, datasets)
    
    # 4. 打印统计
    calculate_and_print(results_dict)