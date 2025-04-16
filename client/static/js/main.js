const socket = io();

let frame0 = document.getElementById('frame0');
let frame1 = document.getElementById('frame1');
let frame2 = document.getElementById('frame2');
let personName = document.getElementById('personName');
let accessStatus = document.getElementById('accessStatus');

socket.on('frame0', data => {
    frame0.src = 'data:image/jpeg;base64,' + data;
});
socket.on('frame1', data => {
    frame1.src = 'data:image/jpeg;base64,' + data;
});
socket.on('frame2', data => {
    frame2.src = 'data:image/jpeg;base64,' + data;
    window.latestGEI = data;
});

socket.on('status', ({ person = 'Unknown', access }) => {
    personName.textContent = person;
    accessStatus.textContent = access ? 'Allowed' : 'Denied';
    accessStatus.classList.toggle('access-denied', !access);
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