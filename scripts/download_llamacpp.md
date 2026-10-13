# llama.cpp Windows binaries (optional GGUF case study)

The paper's primary quantization path is `optimum-quanto` in PyTorch.
GGUF via llama.cpp is an optional deployment case study.

## Option A: Prebuilt release

1. Open https://github.com/ggerganov/llama.cpp/releases
2. Download a Windows CPU build (e.g. `llama-bxxxx-bin-win-avx2-x64.zip`)
3. Extract to `third_party/llama.cpp/`
4. Set environment variable:
   ```powershell
   $env:LLAMA_CPP = "d:\Abubakar Data\Research\Paper2\third_party\llama.cpp"
   ```
5. Run the generated script from `checkpoints/gguf/convert_gguf.sh` (Git Bash) or translate commands to PowerShell.

## Option B: Skip GGUF

If binaries are unavailable, the PyTorch INT8/INT4 results remain the main benchmark; cite GGUF as future deployment work in Limitations.
