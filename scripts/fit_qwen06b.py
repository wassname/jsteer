"""Fit the real Qwen3-0.6B Jacobian cache used by the notebooks and README.

(authored by Claude)

64 diverse web-text-ish prompts (hardcoded below so the fit is reproducible),
mid-layer band, resumable checkpoint. Cost: 1 forward + ceil(1024/8)=128
backwards per prompt; a few minutes on a 3090.

Run:
    uv run python scripts/fit_qwen06b.py 2>&1 | tee /tmp/claude-1000/jsteer_fit06b.log
"""
from __future__ import annotations

import time

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from jsteer import Jacobian

MODEL = "Qwen/Qwen3-0.6B"
OUT = "artifacts/qwen3-0.6b.jac"
CKPT = "artifacts/qwen3-0.6b.ckpt"
DEVICE = "cuda"
DTYPE = torch.bfloat16

# Claude: 64 prompts varying topic (news/science/food/sport/tech/nature/history/
# finance/fiction/how-to/reviews/forum) and register (formal/casual/instructional).
# Each must tokenize to >17 tokens (jlens skips the first 16 positions and drops
# the final one); asserted below.
PROMPTS = [
    # news / civic
    "The city council voted last night to repair the old stone bridge downtown, a project residents have been requesting for well over a decade.",
    "Election officials confirmed on Tuesday that turnout in the rural districts had reached its highest level since records began in the 1970s.",
    "A water main burst under the high street early this morning, closing three shops and flooding the basement of the public library.",
    "The transit authority announced fare increases starting next spring, citing rising maintenance costs and a shortfall in state funding for the third consecutive year.",
    "Firefighters contained the warehouse blaze within two hours, and investigators say the cause appears to be an electrical fault in the loading bay.",
    # science
    "Scientists have long argued about whether the early universe expanded smoothly or in sudden bursts that left traces we can still measure today.",
    "The research team spent four summers tagging migratory birds along the coastline, building one of the largest datasets of its kind in the region.",
    "A new study suggests that soil bacteria can break down certain plastics far faster than previously thought, though scaling the process remains difficult.",
    "Astronomers using the survey telescope catalogued thousands of previously unknown asteroids, most of them harmless rocks orbiting far beyond Mars.",
    "The geology students measured the strata exposed by the landslide, noting how each layer recorded a different chapter of the valley's ancient climate.",
    "Marine biologists were surprised to find the coral colonies recovering faster in the shaded lagoons than in the brightly lit outer reef.",
    # food / cooking
    "Learning to cook well takes patience more than talent, a willingness to taste often, to fail a few times, and to pay attention to detail.",
    "Start by browning the onions slowly in butter over low heat; rushing this step is the most common reason the soup turns out flat.",
    "The bakery on Fifth Street still makes its rye bread the old way, with a starter the owner claims has been alive since the fifties.",
    "If your dough refuses to rise, check the temperature of the water you used, because yeast dies quickly above fifty degrees Celsius.",
    "We tried the new noodle place near the station last weekend, and honestly the broth alone was worth the forty minute wait in line.",
    # sport
    "After months of training, the runners lined up at dawn, breath fogging in the cold air, waiting nervously for the starting gun to fire.",
    "The home side equalised deep in injury time, sending the away fans silent and the commentators scrambling for superlatives they had already used.",
    "Her backhand had always been the weaker shot, so she spent the entire off-season rebuilding it with a coach known for his brutal drills.",
    "The climbing gym gets crowded after work hours, but early on Sunday mornings you can have the whole bouldering wall to yourself.",
    # tech / software
    "Software written in a hurry tends to accumulate small mistakes that hide quietly until, one ordinary afternoon, they surface all at once together.",
    "The outage began when a routine certificate renewal failed silently, and within an hour half the company's internal dashboards were unreachable.",
    "Before you file a bug report, please check whether the issue reproduces on the latest release, and include the full error message and stack trace.",
    "Our team migrated the billing service to the new queue system last quarter, and the pager has been remarkably quiet ever since.",
    "The laptop's battery life is excellent and the keyboard is pleasant, but the webcam remains as grainy as the one on the previous model.",
    "Version control feels like bureaucracy until the first time it saves you, after which committing early and often becomes a reflex.",
    # nature / weather
    "The weather this morning was cold and grey, so I made a large pot of coffee and sat by the window watching the rain fall.",
    "By late August the meadow behind the house turns gold and dry, and grasshoppers scatter ahead of you with every step through the grass.",
    "The first frost arrived weeks earlier than usual this year, catching the gardeners with half the tomato crop still green on the vine.",
    "Wolves returned to the northern valley about a decade ago, and the ripple effects on the deer and the riverbanks are still being studied.",
    "Fog settled over the harbour before sunrise, muffling the bells of the fishing boats as they felt their way out past the breakwater.",
    # history
    "My grandmother used to tell stories about growing up on a small farm, where every season brought a different kind of hard and honest work.",
    "The old mill by the river operated for nearly two centuries before the floods of 1953 finally destroyed the wheel and silted the race.",
    "Medieval copyists made errors just as modern typists do, and scholars can trace manuscript lineages by following those mistakes across the centuries.",
    "The railway reached the town in 1884, and within a generation the wagon roads over the pass had grassed over almost completely.",
    "Archaeologists excavating the harbour district found ledgers recording grain shipments, tax disputes, and one memorable complaint about a dishonest rope merchant.",
    # finance / work
    "The quarterly report showed modest growth in the services division, offset by continued weakness in hardware sales across the European market.",
    "Most household budgets fail not because of large purchases but because of small recurring subscriptions nobody remembers signing up for.",
    "She negotiated the lease renewal down by twelve percent simply by arriving with printouts of comparable rents from the same block.",
    "The startup burned through its first funding round in fourteen months, mostly on an office it did not need and hires it made too early.",
    "Remote work suits some teams wonderfully and hollows out others, and the difference usually comes down to how deliberately they communicate in writing.",
    # fiction-ish / narrative
    "The lighthouse keeper kept a journal for forty years, and the entries grew shorter each winter until they were single words: wind, rain, waiting.",
    "She found the letter tucked inside a secondhand atlas, addressed to a street that, as far as she could tell, had never existed.",
    "The train slowed somewhere between stations and stopped entirely, and for twenty minutes the passengers listened to rain drumming on the roof.",
    "Every house on the street had a story, but the blue one on the corner had three, and none of them agreed with each other.",
    "He repaired watches in the back room of the shop, surrounded by tiny drawers whose labels had worn away decades before he arrived.",
    # how-to / instructional
    "To bleed a radiator, turn off the heating first, hold a cloth under the valve, and open it slowly until water replaces the hissing air.",
    "When packing for a long hike, weigh everything, because ounces you ignore in the kitchen become pounds you curse on the switchbacks.",
    "Sharpen the chisel before each session; a dull edge forces you to push harder, and pushing harder is how fingers end up stitched.",
    "Back up your photos in two places, one of them offsite, because the drive that fails is always the one holding the originals.",
    "Plant the garlic in October, mulch it well, and then do absolutely nothing until the scapes curl in early summer.",
    # reviews / opinion
    "The novel's middle section drags through two hundred pages of committee meetings, but the final chapters justify nearly all of the patience required.",
    "This vacuum is louder than our old one and the cord is shorter, but it picks up pet hair the old machine just pushed around.",
    "The museum's new wing is airy and generous with natural light, though the signage seems designed for people who already know the collection.",
    "I expected the sequel to coast on the original's charm, and instead it quietly fixed almost every complaint I had scribbled down years ago.",
    # forum / casual
    "Does anyone else's cat sprint around the apartment at exactly three in the morning, or have I somehow adopted a very small racehorse?",
    "Update on the sourdough saga: loaf number nine finally has an open crumb, and I owe it all to a colder proof and a hotter oven.",
    "We drove the coast road in a rented van last spring, sleeping at trailheads and eating far too many gas station pastries, and I regret nothing.",
    "My neighbour lends me his ladder every autumn and refuses all payment, so every winter a tray of baklava mysteriously appears on his porch.",
    "Honest question for the group: how do you keep houseplants alive in a north-facing flat, because I have failed four times running.",
    # essay / reflective
    "There is a particular quiet that settles over a library reading room in the late afternoon, part paper, part dust, part concentration.",
    "Teaching a child to ride a bicycle teaches the adult something too, mostly about when to hold on and when to quietly let go.",
    "Cities remember their old rivers even after burying them; the streets still dip and curve along channels no map has shown for a century.",
    "Most advice about writing reduces to the same two acts, performed in alternation: sit down and begin, then cut what you did not need.",
]


def main() -> None:
    assert len(PROMPTS) == 64, len(PROMPTS)
    logger.info(f"loading {MODEL} ({DTYPE}) on {DEVICE}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEVICE).eval()

    # SHOULD: every prompt >17 tokens so jlens (skip_first=16, final dropped)
    # has >=1 valid position. ELSE fit raises on that prompt.
    n_tok = [len(tok(p, add_special_tokens=True).input_ids) for p in PROMPTS]
    logger.info(f"prompt token counts: min={min(n_tok)} max={max(n_tok)}")
    assert min(n_tok) >= 18, f"shortest prompt has {min(n_tok)} tokens"

    t0 = time.monotonic()
    jac = Jacobian.fit(model, tok, PROMPTS, layers=(0.3, 0.9), dim_batch=8,
                       max_seq_len=128, checkpoint_path=CKPT)
    wall = time.monotonic() - t0
    logger.info(f"fit wall-time: {wall:.1f}s ({wall/60:.1f} min) for {len(PROMPTS)} prompts")

    jac.save(OUT)
    logger.info(f"saved {OUT}: {jac!r}")
    # SHOULD: load round-trip returns the same layers. ELSE save/load wiring bug.
    assert Jacobian.load(OUT).layers == jac.layers


if __name__ == "__main__":
    main()
