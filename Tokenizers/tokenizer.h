#include <string>
#include <vector>
#include <unordered_map>
#include <memory>


class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    // Abstract methods to be implemented by specific tokenizers
    virtual std::vector<int> encode(const std::string& text) = 0;
    virtual std::string decode(const std::vector<int>& ids) = 0;

    virtual int token_to_id(const std::string& token) const = 0;
    virtual std::string id_to_token(int id) const = 0;

    // Special token accessor virtual interfaces
    virtual int pad_token_id() const = 0;
    virtual int eos_token_id() const = 0;
    virtual int unk_token_id() const = 0;
};
