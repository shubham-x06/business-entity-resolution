import time

n_tokens = set(["the", "big", "company", "inc"])
a_tokens = set(["123", "main", "st"])
c_n_tokens_pre = set(["the", "big", "company", "ltd"])
c_a_tokens_pre = set(["123", "main", "street"])
c_n_tuple = tuple(["the", "big", "company", "ltd"])
c_a_tuple = tuple(["123", "main", "street"])

# Approach 1: split per query (old)
start = time.time()
for _ in range(100000):
    c_n = "the big company ltd"
    c_a = "123 main street"
    c_n_tokens = set(c_n.split()) if c_n else set()
    c_a_tokens = set(c_a.split()) if c_a else set()
    n_union = len(n_tokens | c_n_tokens)
    a_union = len(a_tokens | c_a_tokens)
    n_sim = len(n_tokens & c_n_tokens) / n_union if n_union else 0.0
    a_sim = len(a_tokens & c_a_tokens) / a_union if a_union else 0.0
print(f"Old (split+set): {time.time()-start:.4f}s")

# Approach 2: precomputed set (my fix)
start = time.time()
for _ in range(100000):
    c_n_tokens = c_n_tokens_pre
    c_a_tokens = c_a_tokens_pre
    n_union = len(n_tokens | c_n_tokens)
    a_union = len(a_tokens | c_a_tokens)
    n_sim = len(n_tokens & c_n_tokens) / n_union if n_union else 0.0
    a_sim = len(a_tokens & c_a_tokens) / a_union if a_union else 0.0
print(f"New 1 (precomputed set | &): {time.time()-start:.4f}s")

# Approach 3: math union, precomputed set
start = time.time()
len_n = len(n_tokens)
len_a = len(a_tokens)
for _ in range(100000):
    c_n_tokens = c_n_tokens_pre
    c_a_tokens = c_a_tokens_pre
    n_inter = len(n_tokens.intersection(c_n_tokens))
    a_inter = len(a_tokens.intersection(c_a_tokens))
    n_union = len_n + len(c_n_tokens) - n_inter
    a_union = len_a + len(c_a_tokens) - a_inter
    n_sim = n_inter / n_union if n_union else 0.0
    a_sim = a_inter / a_union if a_union else 0.0
print(f"New 2 (precomputed set intersection): {time.time()-start:.4f}s")

# Approach 4: tuple precomputed
start = time.time()
for _ in range(100000):
    c_n_tokens = c_n_tuple
    c_a_tokens = c_a_tuple
    n_inter = len(n_tokens.intersection(c_n_tokens))
    a_inter = len(a_tokens.intersection(c_a_tokens))
    n_union = len_n + len(c_n_tokens) - n_inter
    a_union = len_a + len(c_a_tokens) - a_inter
    n_sim = n_inter / n_union if n_union else 0.0
    a_sim = a_inter / a_union if a_union else 0.0
print(f"New 3 (precomputed tuple intersection): {time.time()-start:.4f}s")

