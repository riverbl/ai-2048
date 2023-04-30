#!/bin/sh
cargo clippy -- -W clippy::pedantic -W clippy::nursery -A clippy::cast-possible-wrap
