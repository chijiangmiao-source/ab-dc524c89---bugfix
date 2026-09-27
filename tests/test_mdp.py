"""Unit tests for the exact reachability audit.

The brute-force cross-check enumerates every memoryless policy and solves
its Markov chain exactly with rational Gaussian elimination.  That
machinery lives ONLY in the tests; the service itself never enumerates
policies, never iterates values to a tolerance and never uses floats.
"""

import json
import random
import unittest
from fractions import Fraction

from app import mdp


def build(payload):
    """Validate a model dict, asserting success, and return the Model."""
    return mdp.load_and_validate(json.dumps(payload))


def audit(payload):
    return mdp.audit(build(payload))


def thirds_model():
    return {
        "states": ["s0", "r", "l"],
        "start": "s0",
        "terminals": {"rescued": ["r"], "lost": ["l"]},
        "actions": {
            "s0": {
                "go": {"r": "1/3", "l": "2/3"},
            }
        },
    }


def three_way_relay_model():
    """Deep-sea buoy: three actions at the start state, all with exact
    expected value 2/3.  a1 rescues directly with probability 1/2 and
    falls back to a 1/3-success relay with probability 1/2; a2 and a3
    both commit to a 2/3-success relay."""
    return {
        "states": ["s0", "low", "hi", "r", "l"],
        "start": "s0",
        "terminals": {"rescued": ["r"], "lost": ["l"]},
        "actions": {
            "s0": {
                "a1": {"r": "1/2", "low": "1/2"},
                "a2": {"hi": "1"},
                "a3": {"hi": "1"},
            },
            "low": {"relay": {"r": "1/3", "l": "2/3"}},
            "hi": {"relay": {"r": "2/3", "l": "1/3"}},
        },
    }


def closed_loop_model():
    return {
        "states": ["s0", "k1", "k2", "r", "l"],
        "start": "s0",
        "terminals": {"rescued": ["r"], "lost": ["l"]},
        "actions": {
            "s0": {"drift": {"k1": "1"}},
            "k1": {"spin": {"k2": "1"}},
            "k2": {"spin": {"k1": "1"}},
        },
    }


def assert_certificates_recomputable(testcase, result):
    """Recompute every certificate decision from the state values and the
    per-action expectations, and check it against the emitted certificate:
    optimal/selected flags, the canonical action and every exclusion reason
    must follow from ``state_values`` and ``expected_value`` alone."""
    testcase.assertEqual(result["status"], "ok")
    values = result["state_values"]
    for cert in result["certificates"]:
        value = Fraction(values[cert["state"]])
        testcase.assertEqual(Fraction(cert["value"]), value)
        testcase.assertEqual(cert["value_num"], value.numerator)
        testcase.assertEqual(cert["value_den"], value.denominator)
        entries = cert["actions"]
        testcase.assertTrue(entries)
        optimal = [e["action"] for e in entries if e["optimal"]]
        testcase.assertTrue(optimal, "state %s has no optimal action"
                            % cert["state"])
        least = min(optimal)
        testcase.assertEqual(cert["canonical_action"], least)
        testcase.assertEqual(result["canonical_policy"][cert["state"]], least)
        for entry in entries:
            ev = Fraction(entry["expected_value"])
            testcase.assertEqual(entry["expected_value_num"], ev.numerator)
            testcase.assertEqual(entry["expected_value_den"], ev.denominator)
            testcase.assertEqual(entry["selected"], entry["action"] == least)
            if entry["optimal"]:
                testcase.assertEqual(ev, value)
                reason = (
                    "selected_lexicographically_least_optimal"
                    if entry["action"] == least
                    else "optimal_but_identifier_not_lexicographically_least"
                )
            elif ev < value:
                reason = "expected_value_below_state_value"
            else:
                testcase.assertEqual(ev, value)
                reason = "equal_value_but_no_optimal_continuation"
            testcase.assertEqual(entry["reason"], reason)


class RequiredScenariosTest(unittest.TestCase):
    def test_must_rescue(self):
        result = audit(
            {
                "states": ["s0", "r", "l"],
                "start": "s0",
                "terminals": {"rescued": ["r"], "lost": ["l"]},
                "actions": {"s0": {"go": {"r": "1"}}},
            }
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["max_rescue_probability"], "1")
        self.assertEqual(result["canonical_policy"], {"s0": "go"})
        cert = result["certificates"][0]
        self.assertEqual(cert["state"], "s0")
        self.assertEqual(cert["value"], "1")
        self.assertEqual(cert["actions"][0]["expected_value"], "1")
        self.assertTrue(cert["actions"][0]["selected"])

    def test_thirds_exact_fraction(self):
        result = audit(thirds_model())
        self.assertEqual(result["max_rescue_probability"], "1/3")
        self.assertEqual(result["max_rescue_probability_num"], 1)
        self.assertEqual(result["max_rescue_probability_den"], 3)
        cert = result["certificates"][0]
        self.assertEqual(cert["value"], "1/3")
        self.assertEqual(cert["actions"][0]["expected_value"], "1/3")

    def test_equal_value_actions_pick_lexicographically_least(self):
        payload = {
            "states": ["s0", "r", "l"],
            "start": "s0",
            "terminals": {"rescued": ["r"], "lost": ["l"]},
            "actions": {
                "s0": {
                    "zeta": {"r": "1/2", "l": "1/2"},
                    "alpha": {"r": "1/2", "l": "1/2"},
                    "mid": {"r": "1/2", "l": "1/2"},
                }
            },
        }
        result = audit(payload)
        self.assertEqual(result["canonical_policy"], {"s0": "alpha"})
        cert = result["certificates"][0]
        expected_values = {a["action"]: a["expected_value"] for a in cert["actions"]}
        self.assertEqual(expected_values, {"alpha": "1/2", "mid": "1/2", "zeta": "1/2"})
        for entry in cert["actions"]:
            self.assertTrue(entry["optimal"])
            self.assertEqual(entry["selected"], entry["action"] == "alpha")

    def test_closed_loop_zero_rescue(self):
        result = audit(closed_loop_model())
        self.assertEqual(result["max_rescue_probability"], "0")
        self.assertEqual(result["zero_value_states"], ["k1", "k2", "s0"])
        self.assertEqual(result["terminal_unreachable_states"], ["k1", "k2", "s0"])
        values = {c["state"]: c["value"] for c in result["certificates"]}
        self.assertEqual(values, {"k1": "0", "k2": "0", "s0": "0"})

    def test_three_way_equal_value_relay(self):
        # a1: 1/2 rescued directly + 1/2 low relay (1/3) -> 2/3 exactly;
        # a2, a3: hi relay (2/3) -> 2/3 exactly.  All three actions are
        # truly optimal and the least identifier must be selected.
        result = audit(three_way_relay_model())
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["max_rescue_probability"], "2/3")
        self.assertEqual(result["max_rescue_probability_num"], 2)
        self.assertEqual(result["max_rescue_probability_den"], 3)
        self.assertEqual(result["state_values"]["s0"], "2/3")
        self.assertEqual(result["state_values"]["low"], "1/3")
        self.assertEqual(result["state_values"]["hi"], "2/3")
        self.assertEqual(result["canonical_policy"]["s0"], "a1")
        cert = {c["state"]: c for c in result["certificates"]}["s0"]
        self.assertEqual(cert["value"], "2/3")
        self.assertEqual(cert["canonical_action"], "a1")
        by_action = {a["action"]: a for a in cert["actions"]}
        self.assertEqual(sorted(by_action), ["a1", "a2", "a3"])
        for name, entry in by_action.items():
            self.assertEqual(entry["expected_value"], "2/3")
            self.assertEqual(entry["expected_value_num"], 2)
            self.assertEqual(entry["expected_value_den"], 3)
            self.assertTrue(entry["optimal"])
            self.assertEqual(entry["selected"], name == "a1")
        self.assertEqual(by_action["a1"]["reason"],
                         "selected_lexicographically_least_optimal")
        self.assertEqual(by_action["a2"]["reason"],
                         "optimal_but_identifier_not_lexicographically_least")
        self.assertEqual(by_action["a3"]["reason"],
                         "optimal_but_identifier_not_lexicographically_least")

    def test_input_order_does_not_change_result(self):
        # Shuffling the entry order of states, action blocks, actions and
        # transition items must not change any conclusion.
        rng = random.Random(20260927)
        for model in (three_way_relay_model(), closed_loop_model()):
            base = audit(model)
            for _ in range(25):
                payload = json.loads(json.dumps(model))
                states = payload["states"]
                payload["states"] = rng.sample(states, len(states))
                blocks = list(payload["actions"].items())
                rng.shuffle(blocks)
                new_actions = {}
                for state, amap in blocks:
                    action_items = list(amap.items())
                    rng.shuffle(action_items)
                    new_amap = {}
                    for action, dist in action_items:
                        dist_items = list(dist.items())
                        rng.shuffle(dist_items)
                        new_amap[action] = dict(dist_items)
                    new_actions[state] = new_amap
                payload["actions"] = new_actions
                self.assertEqual(audit(payload), base)

    def test_start_terminal(self):
        result = audit(
            {
                "states": ["s0", "r"],
                "start": "r",
                "terminals": {"rescued": ["r"], "lost": []},
                "actions": {"s0": {"go": {"r": "1"}}},
            }
        )
        self.assertEqual(result["max_rescue_probability"], "1")


class SolverCorrectnessTest(unittest.TestCase):
    def test_bellman_equal_but_not_optimal_is_excluded(self):
        # a2 is Bellman-equal (its expected value equals the state value)
        # but committing to it traps the process in the s1<->s2 loop, so
        # it must NOT be reported as optimal.
        payload = {
            "states": ["s1", "s2", "r", "l"],
            "start": "s1",
            "terminals": {"rescued": ["r"], "lost": ["l"]},
            "actions": {
                "s1": {"a1": {"s2": "1"}},
                "s2": {
                    "a2": {"s1": "1"},
                    "b2": {"r": "1/2", "l": "1/2"},
                },
            },
        }
        result = audit(payload)
        self.assertEqual(result["max_rescue_probability"], "1/2")
        self.assertEqual(result["canonical_policy"], {"s1": "a1", "s2": "b2"})
        cert = {c["state"]: c for c in result["certificates"]}["s2"]
        by_action = {a["action"]: a for a in cert["actions"]}
        self.assertEqual(by_action["a2"]["expected_value"], "1/2")
        self.assertFalse(by_action["a2"]["optimal"])
        self.assertEqual(by_action["a2"]["reason"],
                         "equal_value_but_no_optimal_continuation")
        self.assertTrue(by_action["b2"]["optimal"])
        self.assertTrue(by_action["b2"]["selected"])

    def test_suboptimal_action_reason(self):
        payload = {
            "states": ["s0", "r", "l"],
            "start": "s0",
            "terminals": {"rescued": ["r"], "lost": ["l"]},
            "actions": {
                "s0": {
                    "good": {"r": "2/3", "l": "1/3"},
                    "bad": {"r": "1/3", "l": "2/3"},
                }
            },
        }
        result = audit(payload)
        self.assertEqual(result["max_rescue_probability"], "2/3")
        cert = result["certificates"][0]
        by_action = {a["action"]: a for a in cert["actions"]}
        self.assertEqual(by_action["bad"]["reason"],
                         "expected_value_below_state_value")
        self.assertFalse(by_action["bad"]["optimal"])

    def test_multistep_exact_fraction(self):
        # s0 -> s1 (1/2) / l (1/2); s1 -> r (2/3) / s0 (1/3)
        # v0 = v1/2, v1 = 2/3 + v0/3  =>  v0 = 2/5, v1 = 4/5
        payload = {
            "states": ["s0", "s1", "r", "l"],
            "start": "s0",
            "terminals": {"rescued": ["r"], "lost": ["l"]},
            "actions": {
                "s0": {"go": {"s1": "1/2", "l": "1/2"}},
                "s1": {"go": {"r": "2/3", "s0": "1/3"}},
            },
        }
        result = audit(payload)
        self.assertEqual(result["max_rescue_probability"], "2/5")
        values = {c["state"]: c["value"] for c in result["certificates"]}
        self.assertEqual(values["s1"], "4/5")

    def test_self_loop_with_exit(self):
        # v = 1/3 * 1 + 2/3 * v  =>  v = 1
        payload = {
            "states": ["s0", "r", "l"],
            "start": "s0",
            "terminals": {"rescued": ["r"], "lost": ["l"]},
            "actions": {"s0": {"persist": {"r": "1/3", "s0": "2/3"}}},
        }
        result = audit(payload)
        self.assertEqual(result["max_rescue_probability"], "1")

    def test_certificates_recomputable_from_values_and_expectations(self):
        # Every optimal/selected flag and every exclusion reason must be
        # recomputable from the state values and per-action expectations.
        payloads = [
            three_way_relay_model(),
            closed_loop_model(),
            thirds_model(),
            {  # must-rescue
                "states": ["s0", "r", "l"],
                "start": "s0",
                "terminals": {"rescued": ["r"], "lost": ["l"]},
                "actions": {"s0": {"go": {"r": "1"}}},
            },
            {  # truly suboptimal action
                "states": ["s0", "r", "l"],
                "start": "s0",
                "terminals": {"rescued": ["r"], "lost": ["l"]},
                "actions": {
                    "s0": {
                        "good": {"r": "2/3", "l": "1/3"},
                        "bad": {"r": "1/3", "l": "2/3"},
                    }
                },
            },
            {  # Bellman-equal action without an optimal continuation
                "states": ["s1", "s2", "r", "l"],
                "start": "s1",
                "terminals": {"rescued": ["r"], "lost": ["l"]},
                "actions": {
                    "s1": {"a1": {"s2": "1"}},
                    "s2": {
                        "a2": {"s1": "1"},
                        "b2": {"r": "1/2", "l": "1/2"},
                    },
                },
            },
        ]
        for payload in payloads:
            assert_certificates_recomputable(self, audit(payload))


class ValidationTest(unittest.TestCase):
    def assert_error(self, payload, code, raw=None):
        with self.assertRaises(mdp.AuditError) as ctx:
            mdp.load_and_validate(raw if raw is not None else json.dumps(payload))
        codes = [e["code"] for e in ctx.exception.errors]
        self.assertIn(code, codes)
        for e in ctx.exception.errors:
            self.assertTrue(e["path"].startswith("$"))
            self.assertTrue(e["message"])
        return ctx.exception.errors

    def test_unknown_transition_target(self):
        payload = thirds_model()
        payload["actions"]["s0"]["go"]["ghost"] = "1/3"
        payload["actions"]["s0"]["go"]["r"] = "1/3"
        payload["actions"]["s0"]["go"]["l"] = "1/3"
        errors = self.assert_error(payload, "UNKNOWN_STATE")
        err = [e for e in errors if e["code"] == "UNKNOWN_STATE"][0]
        self.assertIn("ghost", err["path"])

    def test_unknown_start(self):
        payload = thirds_model()
        payload["start"] = "nowhere"
        self.assert_error(payload, "UNKNOWN_STATE")

    def test_missing_actions(self):
        payload = thirds_model()
        payload["states"].append("dangling")
        self.assert_error(payload, "MISSING_ACTIONS")

    def test_terminal_with_actions(self):
        payload = thirds_model()
        payload["actions"]["r"] = {"stay": {"r": "1"}}
        self.assert_error(payload, "TERMINAL_HAS_ACTIONS")

    def test_probability_sum_not_one(self):
        payload = thirds_model()
        payload["actions"]["s0"]["go"] = {"r": "1/2", "l": "1/3"}
        errors = self.assert_error(payload, "PROBABILITY_SUM")
        err = [e for e in errors if e["code"] == "PROBABILITY_SUM"][0]
        self.assertIn("5/6", err["message"])

    def test_non_reduced_fraction(self):
        payload = thirds_model()
        payload["actions"]["s0"]["go"] = {"r": "2/6", "l": "4/6"}
        self.assert_error(payload, "INVALID_PROBABILITY")

    def test_float_probability_rejected(self):
        raw = json.dumps(thirds_model()).replace('"1/3"', "0.5").replace(
            '"2/3"', '"1/2"')
        self.assert_error({}, "INVALID_PROBABILITY", raw=raw)

    def test_duplicate_state(self):
        payload = thirds_model()
        payload["states"].append("s0")
        self.assert_error(payload, "DUPLICATE_STATE")

    def test_duplicate_json_key(self):
        raw = (
            '{"states": ["s0", "r", "l"], "start": "s0",'
            ' "terminals": {"rescued": ["r"], "lost": ["l"]},'
            ' "actions": {"s0": {"go": {"r": "1/3", "r": "1/3", "l": "1/3"}}}}'
        )
        self.assert_error({}, "DUPLICATE_KEY", raw=raw)

    def test_terminal_overlap(self):
        payload = thirds_model()
        payload["terminals"]["lost"].append("r")
        self.assert_error(payload, "DUPLICATE_TERMINAL")

    def test_invalid_json(self):
        with self.assertRaises(mdp.AuditError) as ctx:
            mdp.load_and_validate("{not json")
        self.assertEqual(ctx.exception.errors[0]["code"], "INVALID_JSON")


# ---------------------------------------------------------------------------
# Brute-force cross-check (test-only): enumerate all memoryless policies and
# solve each induced Markov chain exactly.
# ---------------------------------------------------------------------------


def _policy_values(model, policy):
    nonterm = [s for s in model.states
               if s not in model.rescued and s not in model.lost]
    # states that can reach a rescued terminal under this policy
    reach = set(model.rescued)
    changed = True
    while changed:
        changed = False
        for s in nonterm:
            if s not in reach and any(
                t in reach for t in model.actions[s][policy[s]]
            ):
                reach.add(s)
                changed = True
    solvable = [s for s in nonterm if s in reach]
    idx = {s: i for i, s in enumerate(solvable)}
    n = len(solvable)
    # (I - Q) x = beta via exact Gaussian elimination
    mat = [[Fraction(0)] * (n + 1) for _ in range(n)]
    for s in solvable:
        i = idx[s]
        mat[i][i] = Fraction(1)
        for t, p in model.actions[s][policy[s]].items():
            if t in model.rescued:
                mat[i][n] += p
            elif t in idx:
                mat[i][idx[t]] -= p
    for col in range(n):
        piv = next(r for r in range(col, n) if mat[r][col] != 0)
        mat[col], mat[piv] = mat[piv], mat[col]
        pv = mat[col][col]
        mat[col] = [v / pv for v in mat[col]]
        for r in range(n):
            if r != col and mat[r][col] != 0:
                f = mat[r][col]
                mat[r] = [a - f * b for a, b in zip(mat[r], mat[col])]
    values = {s: (mat[idx[s]][n] if s in idx else Fraction(0)) for s in nonterm}
    for r in model.rescued:
        values[r] = Fraction(1)
    for l in model.lost:
        values[l] = Fraction(0)
    return values


def _random_model(rng):
    n_states = rng.randint(1, 3)
    ctr = ["s%d" % i for i in range(n_states)]
    states = ctr + ["r", "l"]
    actions = {}
    for s in ctr:
        n_actions = rng.randint(1, 3)
        amap = {}
        for k in range(n_actions):
            targets = [t for t in states if rng.random() < 0.7] or ["l"]
            weights = [rng.randint(1, 5) for _ in targets]
            total = sum(weights)
            amap["a%d" % k] = {
                t: str(Fraction(w, total)) for t, w in zip(targets, weights)
            }
        actions[s] = amap
    return {
        "states": states,
        "start": "s0",
        "terminals": {"rescued": ["r"], "lost": ["l"]},
        "actions": actions,
    }


class CrossCheckTest(unittest.TestCase):
    def test_against_brute_force_policy_enumeration(self):
        rng = random.Random(20260926)
        for trial in range(60):
            payload = _random_model(rng)
            model = build(payload)
            result = mdp.solve(model)
            nonterm = [s for s in model.states
                       if s not in model.rescued and s not in model.lost]
            act_lists = [sorted(model.actions[s]) for s in nonterm]

            # enumerate every policy and record its exact value vector
            policies = [[]]
            for acts in act_lists:
                policies = [p + [a] for p in policies for a in acts]
            evaluated = []
            for combo in policies:
                policy = dict(zip(nonterm, combo))
                vals = _policy_values(model, policy)
                evaluated.append((policy, tuple(vals[s] for s in nonterm)))
            best = [Fraction(-1)] * len(nonterm)
            for _, vec in evaluated:
                best = [max(a, b) for a, b in zip(best, vec)]
            for i, s in enumerate(nonterm):
                self.assertEqual(
                    result.values[s], best[i],
                    "trial %d state %s payload %s" % (trial, s, payload),
                )
            # canonical action: least action used by some policy attaining
            # the componentwise optimum everywhere
            optimal_policies = [
                policy for policy, vec in evaluated
                if all(vec[j] == best[j] for j in range(len(nonterm)))
            ]
            self.assertTrue(optimal_policies)
            for i, s in enumerate(nonterm):
                optimal_here = {p[s] for p in optimal_policies}
                self.assertEqual(
                    set(result.optimal_actions[s]), optimal_here,
                    "trial %d state %s" % (trial, s),
                )
                self.assertEqual(result.canonical[s], min(optimal_here))


if __name__ == "__main__":
    unittest.main()
