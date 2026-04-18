# SPDX-License-Identifier: Apache-2.0
package kitelogik.agent_lifecycle_test

import data.kitelogik.agent_lifecycle
import future.keywords.if

# --- agent.spawn tests ---

test_allow_spawn_within_depth_limit if {
	agent_lifecycle.allow with input as {
		"event_type": "agent.spawn",
		"action": "agent.spawn",
		"context": {
			"session_id": "s1",
			"user_role": "admin",
			"session_scopes": ["read", "write"],
			"delegation_depth": 0,
		},
		"requested_capabilities": ["read"],
	}
}

test_allow_spawn_at_depth_2 if {
	agent_lifecycle.allow with input as {
		"event_type": "agent.spawn",
		"action": "agent.spawn",
		"context": {
			"session_id": "s1",
			"user_role": "admin",
			"session_scopes": ["read"],
			"delegation_depth": 2,
		},
		"requested_capabilities": ["read"],
	}
}

test_deny_spawn_exceeding_depth if {
	agent_lifecycle.deny with input as {
		"event_type": "agent.spawn",
		"action": "agent.spawn",
		"context": {
			"session_id": "s1",
			"user_role": "admin",
			"session_scopes": ["read"],
			"delegation_depth": 3,
		},
		"requested_capabilities": ["read"],
	}
}

test_deny_spawn_unauthorized_capabilities if {
	agent_lifecycle.deny with input as {
		"event_type": "agent.spawn",
		"action": "agent.spawn",
		"context": {
			"session_id": "s1",
			"user_role": "worker",
			"session_scopes": ["read"],
			"delegation_depth": 0,
		},
		"requested_capabilities": ["write", "delete"],
	}
}

test_allow_spawn_empty_capabilities if {
	agent_lifecycle.allow with input as {
		"event_type": "agent.spawn",
		"action": "agent.spawn",
		"context": {
			"session_id": "s1",
			"user_role": "worker",
			"session_scopes": ["read"],
			"delegation_depth": 0,
		},
		"requested_capabilities": [],
	}
}

# --- agent.delegate tests ---

test_allow_delegate_within_depth if {
	agent_lifecycle.allow with input as {
		"event_type": "agent.delegate",
		"action": "agent.delegate",
		"context": {
			"session_id": "s1",
			"user_role": "admin",
			"session_scopes": ["read", "write", "approve_refund"],
			"delegation_depth": 0,
		},
		"requested_capabilities": ["read"],
	}
}

test_deny_delegate_exceeding_depth if {
	agent_lifecycle.deny with input as {
		"event_type": "agent.delegate",
		"action": "agent.delegate",
		"context": {
			"session_id": "s1",
			"user_role": "admin",
			"session_scopes": ["read"],
			"delegation_depth": 2,
		},
		"requested_capabilities": ["read"],
	}
}

test_deny_delegate_scope_escalation if {
	agent_lifecycle.deny with input as {
		"event_type": "agent.delegate",
		"action": "agent.delegate",
		"context": {
			"session_id": "s1",
			"user_role": "worker",
			"session_scopes": ["read"],
			"delegation_depth": 0,
		},
		"requested_capabilities": ["write", "delete"],
	}
}

# --- malformed delegation_depth tests ---
#
# These close a structural-ordering bypass. In OPA, null < bool < number, so
# `null <= 2` evaluates to TRUE — a spawn/delegate event with a null or missing
# delegation_depth would silently satisfy the depth cap and be allowed at any
# depth. The is_number guard on allow rules plus the explicit non-number deny
# rules close this. These tests lock the behaviour in place.

test_deny_spawn_null_depth if {
	agent_lifecycle.deny with input as {
		"event_type": "agent.spawn",
		"action": "agent.spawn",
		"context": {
			"session_id": "s1",
			"user_role": "admin",
			"session_scopes": ["read"],
			"delegation_depth": null,
		},
		"requested_capabilities": ["read"],
	}
}

test_no_allow_spawn_null_depth if {
	not agent_lifecycle.allow with input as {
		"event_type": "agent.spawn",
		"action": "agent.spawn",
		"context": {
			"session_id": "s1",
			"user_role": "admin",
			"session_scopes": ["read"],
			"delegation_depth": null,
		},
		"requested_capabilities": ["read"],
	}
}

test_deny_spawn_missing_depth if {
	agent_lifecycle.deny with input as {
		"event_type": "agent.spawn",
		"action": "agent.spawn",
		"context": {
			"session_id": "s1",
			"user_role": "admin",
			"session_scopes": ["read"],
		},
		"requested_capabilities": ["read"],
	}
}

test_deny_spawn_string_depth if {
	agent_lifecycle.deny with input as {
		"event_type": "agent.spawn",
		"action": "agent.spawn",
		"context": {
			"session_id": "s1",
			"user_role": "admin",
			"session_scopes": ["read"],
			"delegation_depth": "0",
		},
		"requested_capabilities": ["read"],
	}
}

test_deny_delegate_null_depth if {
	agent_lifecycle.deny with input as {
		"event_type": "agent.delegate",
		"action": "agent.delegate",
		"context": {
			"session_id": "s1",
			"user_role": "admin",
			"session_scopes": ["read"],
			"delegation_depth": null,
		},
		"requested_capabilities": ["read"],
	}
}

test_deny_delegate_missing_depth if {
	agent_lifecycle.deny with input as {
		"event_type": "agent.delegate",
		"action": "agent.delegate",
		"context": {
			"session_id": "s1",
			"user_role": "admin",
			"session_scopes": ["read"],
		},
		"requested_capabilities": ["read"],
	}
}

test_deny_delegate_boolean_depth if {
	# bool < number in OPA — `true <= 1` evaluates TRUE without the is_number guard
	agent_lifecycle.deny with input as {
		"event_type": "agent.delegate",
		"action": "agent.delegate",
		"context": {
			"session_id": "s1",
			"user_role": "admin",
			"session_scopes": ["read"],
			"delegation_depth": true,
		},
		"requested_capabilities": ["read"],
	}
}

# --- no-match tests ---

test_no_allow_for_tool_call_event if {
	not agent_lifecycle.allow with input as {
		"event_type": "tool_call",
		"action": "read_file",
		"context": {
			"session_id": "s1",
			"user_role": "admin",
			"session_scopes": ["read"],
			"delegation_depth": 0,
		},
	}
}
