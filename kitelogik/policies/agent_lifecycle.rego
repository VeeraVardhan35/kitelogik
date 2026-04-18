# SPDX-License-Identifier: Apache-2.0
package kitelogik.agent_lifecycle

import future.keywords.every
import future.keywords.if
import future.keywords.in

default allow := false

default deny := false

# --- agent.spawn rules ---

# Allow spawn when within depth limit and capabilities are valid.
# The is_number guard closes a structural-ordering bypass: in OPA,
# null < number, so `null <= 2` evaluates to TRUE and would allow spawn
# at arbitrary depth when delegation_depth is missing or null.
allow if {
	input.event_type == "agent.spawn"
	is_number(input.context.delegation_depth)
	input.context.delegation_depth <= 2
	_capabilities_valid
}

# Deny spawn when delegation depth exceeds limit
deny if {
	input.event_type == "agent.spawn"
	input.context.delegation_depth > 2
}

# Deny spawn with malformed delegation_depth (missing, null, or non-numeric).
# A well-formed governance event always carries an integer depth — SessionContext
# defaults delegation_depth to 0. A null/missing/non-number value signals a
# malformed event, which should hard-deny rather than fall through to HITL.
# object.get with a null default ensures the value is always bound, so
# `not is_number(...)` also catches the missing-key case.
deny if {
	input.event_type == "agent.spawn"
	not is_number(object.get(input.context, "delegation_depth", null))
}

# Deny spawn when requesting capabilities not in session scopes
deny if {
	input.event_type == "agent.spawn"
	not _capabilities_valid
	count(input.requested_capabilities) > 0
}

# --- agent.delegate rules ---

# Allow delegation when within depth limit and requested scopes are subset of parent.
# is_number guard — see spawn rule comment above.
allow if {
	input.event_type == "agent.delegate"
	is_number(input.context.delegation_depth)
	input.context.delegation_depth <= 1
	_delegate_scopes_valid
}

# Deny delegation that would exceed depth limit
deny if {
	input.event_type == "agent.delegate"
	input.context.delegation_depth > 1
}

# Deny delegation with malformed delegation_depth (see spawn rule above)
deny if {
	input.event_type == "agent.delegate"
	not is_number(object.get(input.context, "delegation_depth", null))
}

# Deny delegation with scopes not in parent's session_scopes
deny if {
	input.event_type == "agent.delegate"
	not _delegate_scopes_valid
	count(input.requested_capabilities) > 0
}

# --- helpers ---

# All requested capabilities must be in the session's scopes
_capabilities_valid if {
	every cap in input.requested_capabilities {
		cap in input.context.session_scopes
	}
}

# Delegation: all requested capabilities must be subset of parent scopes
_delegate_scopes_valid if {
	every scope in input.requested_capabilities {
		scope in input.context.session_scopes
	}
}
