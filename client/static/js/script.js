const socket = io();

socket.on('frame0', data => {
    document.getElementById('cam0').src = 'data:image/jpeg;base64,' + data;
});
socket.on('frame1', data => {
    document.getElementById('cam1').src = 'data:image/jpeg;base64,' + data;
});
socket.on('frame2', data => {
    document.getElementById('cam2').src = 'data:image/jpeg;base64,' + data;
    window.latestGEI = data;
});

socket.on('status', ({ person = 'Unknown', access }) => {
    document.getElementById('personName').textContent = person;

    const statusElement = document.getElementById('accessStatus');
    statusElement.textContent = access ? 'Allowed' : 'Denied';
    statusElement.classList.toggle('access-denied', !access);
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