.PHONY: stop clean test demo landing changelog changelog-release

stop:
	@docker compose stop 2>/dev/null || true
	@echo "containers stopped"

# Remove all local SQLite databases — resets dev state to a blank slate.
clean: stop
	@rm -f hitl.db memory.db events.db credentials.db dev/*.db
	@echo "local databases removed — next start will begin with empty state"

# Demo — run governance quickstart (requires OPA: docker compose up -d opa).
demo:
	python quickstart.py

# Landing page dev server with live reload.
landing:
	@pip install livereload -q
	@echo "  Landing page → http://localhost:8099/landing.html"
	@echo "  Edit docs/landing.html — browser reloads on save. Ctrl-C to stop."
	@python scripts/landing_dev.py

test:
	pytest -q

# Preview the Unreleased section that git-cliff will emit at release time.
# No files are modified; it prints to stdout. Useful before tagging.
changelog:
	@pip install git-cliff -q
	@git-cliff --unreleased

# Prepend the next version's section to CHANGELOG.md. Pass VERSION=v0.2.0.
# Review the diff before committing.
#     make changelog-release VERSION=v0.2.0
changelog-release:
	@test -n "$(VERSION)" || (echo "VERSION is required, e.g. make changelog-release VERSION=v0.2.0"; exit 1)
	@pip install git-cliff -q
	@git-cliff --tag $(VERSION) --unreleased --prepend CHANGELOG.md
	@echo "CHANGELOG.md updated — review and commit."
