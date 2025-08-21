# Gemini Assistant - Project Overview

This document provides a summary of the Unexpected-Keyboard project for the Gemini assistant.

## Project Description

Unexpected-Keyboard is a privacy-focused, open-source keyboard for Android. It aims to provide a great user experience while ensuring that user data remains private.

The project is a fork of the AOSP LatinIME keyboard, not AnySoftKeyboard as mentioned in some documents.

## Current Status

The project is under active development. Key features, such as swipe typing, are in a functional but preliminary state. The current swipe-to-type implementation is a placeholder based on a contribution to the AnySoftKeyboard project and is slated to be replaced by a more advanced, machine-learning-based solution in the future.

## My Role

My role is to assist in the development of Unexpected-Keyboard. I will use my tools to analyze the codebase, suggest improvements, and implement changes as requested. I will adhere to the project's conventions and goals.

## Development Workflow

I will be using my own set of tools to modify the codebase. This includes reading and writing files, searching for code, and running shell commands. I will not use the `patch` command mentioned in `CLAUDE.md`.

## Key Technologies

*   **Language:** Kotlin
*   **Platform:** Android
*   **Build Tool:** Gradle

## TODO

- [ ] Implement a weighted confidence model for swipe gesture word prediction.
- [ ] Add a UI in settings to allow users to edit the weights for the confidence model.
- [ ] Incorporate swipe velocity into the scoring algorithm.
- [ ] Implement path-based pruning for candidate word selection.
- [ ] Use geometric heuristics to further prune candidate words.
- [ ] Implement probabilistic key mapping for ambiguity resolution.
- [ ] Integrate a simple n-gram language model to improve word prediction.