const socket = io();

let frame0 = document.getElementById('frame0');
let frame1 = document.getElementById('frame1');
let frame2 = document.getElementById('frame2');

let person = document.getElementById('person');
let access = document.getElementById('access');
let confidence = document.getElementById('confidence');

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
        person.textContent = data.person;
        access.textContent = data.access ? 'Granted' : 'Denied';
        access.classList.toggle('access-denied', !data.access);

        if (data.person === 'Unknown') {
            confidence.textContent = 'N/A';
        } else if (data.confidence !== undefined) {
            confidence.textContent = `${data.confidence}%`;
        } else {
            confidence.textContent = 'Unknown';
        }

    }
});

resetBtn.addEventListener('click', () => {
    socket.emit('reset');
});

saveBtn.addEventListener('click', () => {
    const label = prompt('Enter label for this GEI:');
    if (label) {
        socket.emit('save', { label: label, gei: window.latestGEI });
    }
});