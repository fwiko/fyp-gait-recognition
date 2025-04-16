const socket = io();

let frame0 = document.getElementById('frame0');
let frame1 = document.getElementById('frame1');
let frame2 = document.getElementById('frame2');
let personName = document.getElementById('personName');
let accessStatus = document.getElementById('accessStatus');
let resetBtn = document.getElementById('resetBtn');
let saveBtn = document.getElementById('saveBtn');

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

socket.on('status', (data) => {
    if (data.person) {
        personName.textContent = data.person;
        accessStatus.textContent = data.access ? 'Granted' : 'Denied';
        accessStatus.classList.toggle('access-denied', !data.access);
    }
});

resetBtn.addEventListener('click', () => {
    socket.emit('reset');
});

saveBtn.addEventListener('click', () => {
    const label = prompt('Enter label for this GEI:');
    if (label) {
        socket.emit('save', { label });
    }
});