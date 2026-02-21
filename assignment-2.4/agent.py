import utils
from math import log2
from collections import defaultdict
from client import Agent

class ExitAgent(Agent):
    def __init__(self, run_id, agent_config):
        super().__init__(run_id, agent_config)

        # A new instance is created for each run (i.e. each game).
        # So you can keep a run-specific state.
        # WARNING: If a run was not finished, it will be resumed.
        #          get_action will then be called for the next required action.

        env_string          = agent_config.get("env", "")
        self.is_advanced    = "advanced" in env_string
        
        self.possible_words = []
        self.word_weights   = {}   
        self.possible_lengths = set()
        self.revealed_letters = set()
        self.guessed_letters  = set()

        self.tz_words, self.non_tz_words = utils.load_word_lists("worldcities.csv.bz2")

    def _initialize_possible_words(self):
        feedback = self.percept['feedback']

        revealed_letters_count = sum(1 for c in feedback if c != '-')

        min_len = max(1, revealed_letters_count) 
        max_len = len(feedback)
        
        self.possible_lengths = set(range(min_len, max_len + 1))

        tz_base_words       = [w for w in self.tz_words if len(w) in self.possible_lengths]
        non_tz_base_words   = [w for w in self.non_tz_words if len(w) in self.possible_lengths]

        total_tz        = len(tz_base_words)
        total_non_tz    = len(non_tz_base_words)

        self.word_weights = {}
        if total_tz == 0 and total_non_tz == 0:
            self.possible_words = []
            self.revealed_letters.clear()
            self.guessed_letters.clear()
            return

        if total_tz > 0:
            for w in tz_base_words:
                self.word_weights[w] = 0.5 / total_tz
        if total_non_tz > 0:
            for w in non_tz_base_words:
                self.word_weights[w] = 0.5 / total_non_tz

        self.possible_words = tz_base_words + non_tz_base_words

        self.revealed_letters.clear()
        self.guessed_letters.clear()

    def normalize_word_weights(self):
        total_wt = sum(self.word_weights.get(w, 0.0) for w in self.possible_words)

        if total_wt > 0:
            for w in self.possible_words:
                self.word_weights[w] = self.word_weights[w] / total_wt
        else:
            if self.possible_words:
                uniform = 1.0 / len(self.possible_words)
                for w in self.possible_words:
                    self.word_weights[w] = uniform
        
    def discard_invalid_words(self):
        invalid_guesses     = set([g for g in self.percept['guesses'] if len(g) > 1])
        self.possible_words = list(set(self.possible_words) - invalid_guesses)

    def apply_pattern_match(self):
        
        new_possible = []
        feedback     = self.percept['feedback']
        
        currently_revealed = {c for c in feedback if c != '-'}
        self.revealed_letters.update(currently_revealed)
        max_length = len(feedback)

        last_revealed_index   = max((i for i, ch in enumerate(feedback) if ch != '-'), default=-1) + 1
        self.possible_lengths = set(range(last_revealed_index, max_length + 1))

        absent_letters = self.guessed_letters - self.revealed_letters
        
        for word in self.possible_words:
            if len(word) not in self.possible_lengths:
                continue

            match = all(i < len(word) and word[i] == ch for i, ch in enumerate(feedback) if ch != '-')
            match = match and not any(ltr in word for ltr in absent_letters)

            if match:
                new_possible.append(word)

        self.possible_words = new_possible

    def _update_possible_words(self):
        self.discard_invalid_words()
        self.apply_pattern_match()
        self.normalize_word_weights()

    def calculate_information_gain(self, letter):
        if letter in self.guessed_letters:
            return -float('inf')

        total_weight = sum(self.word_weights.get(w, 0.0) for w in self.possible_words)

        if total_weight <= 0:
            return -float('inf')

        H_before = 0.0
        for w in self.possible_words:
            p = self.word_weights.get(w, 0.0)
            if p > 0:
                H_before -= p * log2(p)

        partitions = defaultdict(list)

        for word in self.possible_words:
            wgt = self.word_weights.get(word, 0.0)
            positions = [i for i, ch in enumerate(word) if ch == letter]
            
            if not positions:
                key = ('absent',)
            else:
                key = tuple(positions)

            partitions[key].append((word, wgt))

        if not partitions:
            return -float('inf')

        H_after = 0.0

        for key, cluster in partitions.items():
            cluster_weight      = sum(w for _, w in cluster)
            prob_of_partition   = cluster_weight / total_weight

            if key[0] == 'absent':
                H_cluster = 0.0
                for _, w in cluster:
                    p = w / cluster_weight
                    if p > 0:
                        H_cluster -= p * log2(p)
                H_after += prob_of_partition * H_cluster

            else:
                if not self.is_advanced:
                    # === Standard Rules ===
                    H_cluster = 0.0
                    for _, w in cluster:
                        p = w / cluster_weight
                        if p > 0:
                            H_cluster -= p * log2(p)
                    H_after += prob_of_partition * H_cluster

                else:
                    # === Advanced Rules ===
                    positions_list = list(key)                    
                    fraction = 1.0 / len(positions_list)

                    subclusters_by_pos = defaultdict(list)

                    for (wrd, wgt) in cluster:
                        for pos in positions_list:
                            subclusters_by_pos[pos].append((wrd, wgt * fraction))

                    cluster_entropy = 0.0
                    for pos, subcluster in subclusters_by_pos.items():
                        subcluster_weight = sum(sw for _, sw in subcluster)
                        pos_prob = subcluster_weight / cluster_weight if cluster_weight > 0 else 0

                        H_sub = 0.0
                        for (_, sw) in subcluster:
                            p_sub = sw / subcluster_weight if subcluster_weight > 0 else 0
                            if p_sub > 0:
                                H_sub -= p_sub * log2(p_sub)

                        cluster_entropy += pos_prob * H_sub

                    H_after += prob_of_partition * cluster_entropy

        return H_before - H_after

    def update_possible_words(self):
        if not self.possible_words:
            self._initialize_possible_words()
        else:
            self._update_possible_words()
    
    def update_guessed_letters(self):
        self.guessed_letters = set([g for g in self.percept['guesses'] if len(g) == 1])

    def get_sorted_candidates(self):
        freq = defaultdict(int)
        for word in self.possible_words:
            for c in set(word):
                if c not in self.guessed_letters:
                    freq[c] += 1
        return sorted(freq.keys(), key=lambda x: -freq[x])

    def get_action(self, percept, request_info):
        self.percept = percept

        self.update_guessed_letters()
        self.update_possible_words()
        
        if len(self.possible_words) == 1:
            return self.possible_words[0]

        if self.possible_words:
            best_word = max(self.possible_words, key=lambda w: self.word_weights[w])
            best_word_prob = self.word_weights[best_word]
            if best_word_prob >= 0.30:
                return best_word

        letter_order = self.get_sorted_candidates()
        candidates   = [c for c in letter_order if c not in self.guessed_letters]
        
        if not candidates:
            return self.possible_words[0] if self.possible_words else "A"

        best_letter = max(candidates, key=lambda x: self.calculate_information_gain(x), 
                          default = next((ltr for ltr in letter_order if ltr not in self.guessed_letters), "A"))	
        
        best_gain   = self.calculate_information_gain(best_letter)

        if best_gain <= 0:
            best_letter = next((c for c in candidates if any(c in w for w in self.possible_words)), 
                               self.possible_words[0] if self.possible_words else "A")

        return best_letter


if __name__ == '__main__':
    import sys, logging
    from client import run

    # You can set the logging level to logging.WARNING or logging.ERROR for less output.
    logging.basicConfig(level=logging.INFO)

    ExitAgent.run(
        agent_config_file=sys.argv[1],
        parallel_runs=True,    # If set to True, the server creates multiple parallel runs and bundles the requests.
        multiprocessing=True,  # Use multiple processes to run multiple agents in parallel.
        run_limit=1000,         # Stop after 1000 runs. Set to 1 for debugging.
    )

    # TIP: If your agent works, consider setting parallel_runs=True and multiprocessing=True.

