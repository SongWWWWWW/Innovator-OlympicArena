import os
from typing import List, Union, Dict, Any, Callable
from vllm import LLM, SamplingParams
from .base_model import BaseModel

class Qwen3(BaseModel):
    def __init__(
        self, 
        model_path: str, 
        tensor_parallel_size: int = 8, 
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 32768,
        **kwargs
    ):
        """
        :param model_path: 模型路径
        :param tensor_parallel_size: TP并行数（多卡必填）
        """
        super().__init__(model_path, **kwargs) # 保持基类初始化逻辑
        self.model_path = model_path
        
        # 初始化 vLLM 引擎
        self.model = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype="bfloat16",
            enforce_eager=False,
        )

    def generate_single(
        self, 
        prompt: str, 
        max_new_tokens: int = 8196,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop_words: List[str] = None,
        **kwargs
    ) -> str:
        """
        基类要求的单条推理接口
        """
        sampling_params = SamplingParams(
            n=1,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            stop=stop_words,
        )
        outputs = self.model.generate([prompt], sampling_params, use_tqdm=False)
        return outputs[0].outputs[0].text

    def generate_batch(
        self, 
        prompts: List[str], 
        max_new_tokens: int = 8196,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_workers: int = None,      # 保持接口一致，但 vLLM 不需要线程池
        save_callback: Callable = None, # 用于每条结果生成后的回调保存
        stop_words: List[str] = None,
        **kwargs
    ) -> List[str]:
        """
        适配 BaseModel 的批量推理
        """
        sampling_params = SamplingParams(
            n=1,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            stop=stop_words,
        )
        
        # 这里的最快方式是不使用多线程，直接交给 vLLM 的 scheduler
        outputs = self.model.generate(prompts, sampling_params, use_tqdm=True)
        
        final_results = []
        for i, output in enumerate(outputs):
            res_text = output.outputs[0].text
            final_results.append(res_text)
            
            # 如果提供了保存回调，则模仿线程池模式在结果产生后立即触发
            if save_callback:
                save_callback(i, res_text, False)
        
        return final_results

    def generate_batch_N(
        self, 
        prompts: List[str], 
        n: int, 
        max_new_tokens: int = 32768,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_workers: int = None,      # 保持接口一致
        save_callback: Callable = None, 
        stop_words: List[str] = None,
        **kwargs
    ) -> List[List[str]]:
        """
        适配 BaseModel 的 Pass@k 批量推理
        """
        sampling_params = SamplingParams(
            n=n, # 并行采样 n 个
            temperature=max(temperature, 0.1),
            top_p=top_p,
            max_tokens=max_new_tokens,
            stop=stop_words,
        )

        outputs = self.model.generate(prompts, sampling_params, use_tqdm=True)

        final_results = []
        for i, output in enumerate(outputs):
            # 获取该 prompt 对应的 n 个生成结果
            batch_n_res = [cand.text for cand in output.outputs]
            final_results.append(batch_n_res)
            
            # 触发保存回调
            if save_callback:
                save_callback(i, batch_n_res, True)
        
        return final_results