# frozen_string_literal: true

source "https://rubygems.org"
gemspec

gem "jekyll", ENV["JEKYLL_VERSION"] if ENV["JEKYLL_VERSION"]
gem "kramdown-parser-gfm" if ENV["JEKYLL_VERSION"] == "~> 3.9"

gem "webrick", "~> 1.9"

# Gemas necesarias para Ruby 3.4.0 y superiores
gem 'csv'
gem 'base64'
gem 'bigdecimal'
gem "github-pages", "~> 228", group: :jekyll_plugins