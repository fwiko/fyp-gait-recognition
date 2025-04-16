const socket = io();
const personName = document.getElementById('personName');
const accessStatus = document.getElementById('accessStatus');
const resetBtn = document.getElementById('resetBtn');
const saveBtn = document.getElementById('saveBtn');

for (let i = 0; i < 3; i++) {
    socket.on(`frame${i}`, (data) => {
        document.getElementById(`frame${i}`).src = `data:image/jpeg;base64,${data}`;
    });
}

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