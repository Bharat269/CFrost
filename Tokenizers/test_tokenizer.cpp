#include "simple_tokenizer.cpp"
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

void test_special_tokens_and_seeded_punctuation() {
    cfrost::SimpleTokenizer tok;

    assert(tok.pad_token_id() == 0);
    assert(tok.unk_token_id() == 1);
    assert(tok.eos_token_id() == 2);

    assert(tok.id_to_token(tok.pad_token_id()) == "<pad>");
    assert(tok.id_to_token(tok.unk_token_id()) == "<unk>");
    assert(tok.id_to_token(tok.eos_token_id()) == "<eos>");

    // Verify pre-seeded punctuation gets valid non-unk IDs
    int period_id = tok.token_to_id(".");
    int comma_id = tok.token_to_id(",");
    int bang_id = tok.token_to_id("!");
    assert(period_id != tok.unk_token_id());
    assert(comma_id != tok.unk_token_id());
    assert(bang_id != tok.unk_token_id());
    assert(period_id != comma_id);
    assert(tok.id_to_token(period_id) == ".");
}

void test_encode_words_and_punctuation() {
    cfrost::SimpleTokenizer tok;

    std::string text = "Hello, world! C++ fast-tokenizer.";
    std::vector<int> ids = tok.encode(text);

    // Expected tokens: ["Hello", ",", "world", "!", "C", "+", "+", "fast", "-", "tokenizer", "."]
    assert(ids.size() == 11);

    std::vector<std::string> decoded_tokens = tok.ids_to_tokens(ids);
    assert(decoded_tokens.size() == 11);
    assert(decoded_tokens[0] == "Hello");
    assert(decoded_tokens[1] == ",");
    assert(decoded_tokens[2] == "world");
    assert(decoded_tokens[3] == "!");
    assert(decoded_tokens[4] == "C");
    assert(decoded_tokens[5] == "+");
    assert(decoded_tokens[6] == "+");
    assert(decoded_tokens[7] == "fast");
    assert(decoded_tokens[8] == "-");
    assert(decoded_tokens[9] == "tokenizer");
    assert(decoded_tokens[10] == ".");

    // Repeated words/punctuation should reuse existing IDs rather than duplicating
    std::vector<int> repeat_ids = tok.encode("Hello world.");
    assert(repeat_ids.size() == 3);
    assert(repeat_ids[0] == ids[0]);
    assert(repeat_ids[1] == ids[2]);
    assert(repeat_ids[2] == ids[10]);
}

void test_decode_and_special_tokens() {
    cfrost::SimpleTokenizer tok;

    std::vector<int> ids = tok.encode("Hi, there!");
    assert(tok.decode(ids) == "Hi, there!");

    // Ensure <pad> and <eos> are skipped cleanly during decode
    std::vector<int> with_specials = {tok.pad_token_id()};
    with_specials.insert(with_specials.end(), ids.begin(), ids.end());
    with_specials.push_back(tok.eos_token_id());
    with_specials.push_back(tok.pad_token_id());

    assert(tok.decode(with_specials) == "Hi, there!");
}

void test_batch_lookups_and_bounds() {
    cfrost::SimpleTokenizer tok;
    tok.encode("alpha beta");

    std::vector<std::string> query_tokens = {"alpha", "never_seen_token", "beta"};
    std::vector<int> query_ids = tok.tokens_to_ids(query_tokens);

    assert(query_ids.size() == 3);
    assert(query_ids[0] == tok.token_to_id("alpha"));
    assert(query_ids[1] == tok.unk_token_id());
    assert(query_ids[2] == tok.token_to_id("beta"));

    // Out-of-range IDs should safely fall back to <unk>
    assert(tok.id_to_token(-1) == "<unk>");
    assert(tok.id_to_token(999999) == "<unk>");
}

void test_empty_and_whitespace_only() {
    cfrost::SimpleTokenizer tok;

    assert(tok.encode("").empty());
    assert(tok.encode("   \t\n  ").empty());
    assert(tok.decode({}).empty());
}

int main() {
    test_special_tokens_and_seeded_punctuation();
    test_encode_words_and_punctuation();
    test_decode_and_special_tokens();
    test_batch_lookups_and_bounds();
    test_empty_and_whitespace_only();

    std::cout << "All SimpleTokenizer tests passed!\n";
    return 0;
}