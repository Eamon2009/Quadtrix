#include <torch/script.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <csignal>

// handler for Ctrl+C (stop)
bool running = true;
void handle_sigint(int)
{
      std::cout << "\n\n[Stopped by user]\n";
      running = false;
}

// Load vocab from cleaned.txt
std::map<int, char> build_vocab(const std::string &path)
{
      std::ifstream file(path);
      if (!file.is_open())
      {
            std::cerr << "[Error] Cannot open: " << path << "\n";
            exit(1);
      }
      std::string text((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());

      std::set<char> char_set(text.begin(), text.end());
      std::vector<char> chars(char_set.begin(), char_set.end());
      std::sort(chars.begin(), chars.end());

      std::map<int, char> it;
      for (int i = 0; i < (int)chars.size(); i++)
            it[i] = chars[i];

      return it;
}

int main()
{
      signal(SIGINT, handle_sigint);
      std::string model_path = "../best_model.pt";
      std::string cleaned_path = "../cleaned.txt";

      // Load TorchScript model
      std::cout << "--- Loading Pre-trained Model ---\n";
      torch::jit::script::Module model;
      try
      {
            model = torch::jit::load(model_path, torch::kCPU);
      }
      catch (const c10::Error &e)
      {
            std::cerr << "[Error] Could not load model: " << e.what() << "\n";
            std::cerr << "Make sure you have exported best_model.pt as TorchScript (see export.py)\n";
            return 1;
      }
      model.eval();

      //  vocab
      auto it = build_vocab(cleaned_path);
      std::cout << "[INFO] Vocab size: " << it.size() << "\n";
      std::cout << "[INFO] Device: CPU\n";
      std::cout << "Model loaded. Generating text (Ctrl+C to stop)...\n\n";
      std::cout << std::string(50, '-') << "\n";

      // Inference loop
      const int block_size = 128;
      auto context = torch::zeros({1, 1}, torch::kLong);

      torch::NoGradGuard no_grad;

      while (running)
      {
            // Crop context to block_size
            auto idx_cond = context;
            if (context.size(1) > block_size)
                  idx_cond = context.slice(1, context.size(1) - block_size);

            // Forward pass
            std::vector<torch::jit::IValue> inputs = {idx_cond};
            auto output = model.forward(inputs).toTuple();
            auto logits = output->elements()[0].toTensor();

            // Get last token logits
            logits = logits.select(1, logits.size(1) - 1);

            // Softmax  sample
            auto probs = torch::softmax(logits, -1);
            auto idx_next = torch::multinomial(probs, 1);

            // Append to context
            context = torch::cat({context, idx_next}, 1);
            if (context.size(1) > block_size)
                  context = context.slice(1, context.size(1) - block_size);

            // Decode and print
            int token = idx_next[0][0].item<int>();
            if (it.count(token))
                  std::cout << it[token] << std::flush;
            else
                  std::cout << '?' << std::flush;
      }

      return 0;
}