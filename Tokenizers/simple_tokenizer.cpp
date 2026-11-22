#include "tokenizer.h"

#include <cctype>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>

namespace cfrost {

class SimpleTokenizer : public Tokenizer {
public:
    SimpleTokenizer() {
        // Reserve standard special token IDs upfront
        register_token("<pad>"); // 0
        register_token("<unk>"); // 1
        register_token("<eos>"); // 2

        // Seed common punctuation marks into the dictionary
        const std::string default_punct = ".,!?;:'\"()-[]{}";
        for (char ch : default_punct) {
            register_token(std::string(1, ch));
        }
    }

    ~SimpleTokenizer() override = default;

    std::vector<int> encode(const std::string& text) override {
        std::vector<std::string> tokens = split_words_and_punct(text);

        std::vector<int> ids;
        ids.reserve(tokens.size());
        for (const auto& tok : tokens) {
            // Add new words/punctuation to the dictionary automatically
            // should not happen here
            // this is purely a lookup function
            //if token not found then return <unk> token id
            // int id = token_to_id(tok);
            //will have to check and discuss this whole thing cause gotta
            //freeze the vocab during training to
            // see response to unknown data
            int id  = get_or_add_token(tok);
            ids.push_back(id);
        }
        return ids;
    }

    std::string decode(const std::vector<int>& ids) override {
        std::vector<std::string> tokens = ids_to_tokens(ids);
        std::string result;

        for (size_t i = 0; i < tokens.size(); ++i) {
            const std::string& tok = tokens[i];

            // Skip special tokens during text reconstruction
            if (tok == "<pad>" || tok == "<eos>") {
                continue;
            }

            // Avoid adding a leading space before punctuation marks
            if (!result.empty() && !is_single_punct(tok)) {
                result += ' ';
            }
            result += tok;
        }

        return result;
    }

    int token_to_id(const std::string& token) const override {
        auto it = token_to_id_.find(token);
        if (it != token_to_id_.end()) {
            return it->second;
        }
        return unk_id_;
    }

    std::string id_to_token(int id) const override {
        if (id >= 0 && id < static_cast<int>(id_to_token_.size())) {
            return id_to_token_[id];
        }
        return id_to_token_[unk_id_];
    }

    std::vector<int> tokens_to_ids(const std::vector<std::string>& tokens) {
        std::vector<int> ids;
        ids.reserve(tokens.size());
        for (const auto& tok : tokens) {
            // Add new words/punctuation to the dictionary automatically
            // should not happen here
            // this is purely a lookup function
            //if token not found then return <unk> token id
            // int id = token_to_id(tok);
            //will have to check and discuss this whole thing cause gotta
            //freeze the vocab during training to
            // see response to unknown data
            int id  = token_to_id(tok);
            ids.push_back(id);
        }
        return ids;
    }

    std::vector<std::string> ids_to_tokens(const std::vector<int>& ids) {
        std::vector<std::string> tokens;
        tokens.reserve(ids.size());
        for (int id : ids) {
            std::string token = id_to_token(id);
            std::cout << token << " ";
            tokens.push_back(token);
        }
        return tokens;
    }

    int pad_token_id() const override { return pad_id_; }
    int eos_token_id() const override { return eos_id_; }
    int unk_token_id() const override { return unk_id_; }

private:
    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::string> id_to_token_;

    int pad_id_ = 0;
    int unk_id_ = 1;
    int eos_id_ = 2;

    int register_token(const std::string& token) {
        auto it = token_to_id_.find(token);
        if (it != token_to_id_.end()) {
            return it->second;
        }
        int new_id = static_cast<int>(id_to_token_.size());
        token_to_id_[token] = new_id;
        id_to_token_.push_back(token);
        return new_id;
    }

    int get_or_add_token(const std::string& token) {
        return register_token(token);
    }

    static bool is_punct_char(unsigned char ch) {
        return std::ispunct(ch) != 0;
    }

    static bool is_single_punct(const std::string& tok) {
        return tok.size() == 1 && is_punct_char(static_cast<unsigned char>(tok[0]));
    }

    // Splits on whitespace and separates punctuation characters into standalone tokens
    std::vector<std::string> split_words_and_punct(const std::string& text) const {
        std::vector<std::string> tokens;
        std::istringstream stream(text);
        std::string chunk;

        while (stream >> chunk) {
            std::string current_word;
            for (unsigned char ch : chunk) {
                if (is_punct_char(ch)) {
                    if (!current_word.empty()) {
                        tokens.push_back(current_word);
                        current_word.clear();
                    }
                    tokens.push_back(std::string(1, static_cast<char>(ch)));
                } else {
                    current_word += static_cast<char>(ch);
                }
            }
            if (!current_word.empty()) {
                tokens.push_back(current_word);
            }
        }

        return tokens;
    }
};

} // namespace cfrost