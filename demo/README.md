---
title: AI Chessathon Engine
emoji: ♞
colorFrom: gray
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# AI Chessathon Engine

Play the two engines built for the AI Chessathon: an NNUE evaluation trained on
engine-labelled positions, searched by an alpha-beta search written as numba kernels
over bitboards. Every engine move comes with its search telemetry: depth reached, nodes,
nodes per second, the score, the principal variation, and the node count at every
iteration, so you can see the branching arithmetic that decides a move.

- **Gators (net2)**: the build that played rated games on the platform.
- **net5 (latest)**: the same search with the most recent net.

The first request after a cold start takes about a minute while the kernels compile.
Source, training notes and the write-up: https://github.com/ArnavBagmar/aichess
