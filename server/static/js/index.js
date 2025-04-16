const socket = io();

// Handle access toggle
document.addEventListener('DOMContentLoaded', function () {
    const accessToggle = document.getElementById('accessToggle');
    if (accessToggle) {
        accessToggle.addEventListener('change', function (e) {
            const identityId = this.dataset.identityId;
            const allowAccess = this.checked;

            socket.emit('update-access-rule', {
                identity_id: identityId,
                allow_access: allowAccess
            });
        });
    }

    // Handle identity deletion
    const deleteIdentityBtn = document.getElementById('deleteIdentity');
    if (deleteIdentityBtn) {
        deleteIdentityBtn.addEventListener('click', function () {
            if (confirm('Are you sure you want to delete this identity? This will remove all associated gait samples.')) {
                const identityId = this.dataset.identityId;
                socket.emit('delete-identity', { identity_id: identityId });
            }
        });
    }

    // Handle sample deletion
    document.querySelectorAll('.delete-sample').forEach(button => {
        button.addEventListener('click', function () {
            const sampleId = this.dataset.sampleId;
            if (confirm('Are you sure you want to delete this gait sample?')) {
                socket.emit('delete-gait-sample', { sample_id: sampleId });
            }
        });
    });
});

// Listen for access rule updates
socket.on('access_rule_updated', function (data) {
    if (data.success) {
        const statusElement = document.querySelector('.identity-status');
        if (statusElement) {
            // Update only the text node, preserving the toggle
            const textNode = statusElement.childNodes[0];
            textNode.textContent = data.allow_access ? 'Access Allowed' : 'Access Denied';
            statusElement.className = 'identity-status ' + (data.allow_access ? 'access-allowed' : 'access-denied');
        }
    } else {
        // Revert toggle if update failed
        const toggle = document.getElementById('accessToggle');
        if (toggle) {
            toggle.checked = !toggle.checked;
            alert('Failed to update access rule. Please try again.');
        }
    }
});

// Listen for identity deletion response
socket.on('identity_deleted', function (data) {
    if (data.success) {
        window.location.href = '/';
    } else {
        alert('Failed to delete identity. Please try again.');
    }
});

// Listen for gait sample deletion response
socket.on('gait_sample_deleted', function (data) {
    if (data.success) {
        // Remove the card from the UI
        const card = document.querySelector(`.sample-card[data-sample-id="${data.sample_id}"]`);
        if (card) {
            card.remove();
            // Update the sample count
            const sampleCount = document.querySelector('.gait-samples h3');
            const currentCount = parseInt(sampleCount.textContent.match(/\d+/)[0]);
            sampleCount.textContent = `Gait Samples (${currentCount - 1})`;
        }
    } else {
        alert('Failed to delete gait sample. Please try again.');
    }
});

function reset() {
    socket.emit('reset');
}

function save() {
    const label = prompt("Enter identity label (e.g., John Smith):");
    if (!label || !window.latestGEI) {
        alert("Missing label or GEI.");
        return;
    }

    socket.emit('save', {
        label: label,
        gei: window.latestGEI
    });
}