// static/js/main.js
document.addEventListener('DOMContentLoaded', function () {
    // Function to format dates in a more human-readable format
    function formatTimeAgo(dateString) {
        const date = new Date(dateString);
        const now = new Date();

        const seconds = Math.floor((now - date) / 1000);

        // Less than a minute
        if (seconds < 60) {
            return 'just now';
        }

        // Less than an hour
        const minutes = Math.floor(seconds / 60);
        if (minutes < 60) {
            return `${minutes} minute${minutes > 1 ? 's' : ''} ago`;
        }

        // Less than a day
        const hours = Math.floor(minutes / 60);
        if (hours < 24) {
            return `${hours} hour${hours > 1 ? 's' : ''} ago`;
        }

        // Less than a month
        const days = Math.floor(hours / 24);
        if (days < 30) {
            return `${days} day${days > 1 ? 's' : ''} ago`;
        }

        // Use the date
        return date.toLocaleDateString();
    }

    // Add the current year to the footer
    const footerYear = document.querySelector('footer p');
    if (footerYear) {
        const currentYear = new Date().getFullYear();
        footerYear.textContent = footerYear.textContent.replace(/\d{4}/, currentYear);
    }

    // Format any dates with the 'time-ago' class
    const timeElements = document.querySelectorAll('.time-ago');
    timeElements.forEach(element => {
        const dateString = element.dataset.date;
        if (dateString) {
            element.textContent = formatTimeAgo(dateString);
        }
    });

    // Setup Socket.IO connection for real-time updates if needed
    if (typeof io !== 'undefined') {
        const socket = io();

        socket.on('connect', function () {
            console.log('Connected to server');
        });

        socket.on('identity_update', function (data) {
            console.log('Identity update received:', data);
            // Handle identity updates - refresh data or show notification
            if (window.location.pathname === '/' ||
                window.location.pathname === '/access_rules') {
                // Refresh the page to show updated data
                location.reload();
            }
        });

        socket.on('recognition_event', function (data) {
            // Handle recognition events - show notification
            showNotification(`Recognition: ${data.label}`,
                `Confidence: ${Math.round(data.confidence * 100)}% - 
                            Access: ${data.access_allowed ? 'Allowed' : 'Denied'}`);
        });
    }

    // Notification function
    function showNotification(title, message) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = 'notification';

        const notificationTitle = document.createElement('div');
        notificationTitle.className = 'notification-title';
        notificationTitle.textContent = title;

        const notificationMessage = document.createElement('div');
        notificationMessage.className = 'notification-message';
        notificationMessage.textContent = message;

        const closeButton = document.createElement('button');
        closeButton.className = 'notification-close';
        closeButton.innerHTML = '&times;';
        closeButton.addEventListener('click', function () {
            notification.remove();
        });

        notification.appendChild(closeButton);
        notification.appendChild(notificationTitle);
        notification.appendChild(notificationMessage);

        // Add to document
        document.body.appendChild(notification);

        // Animate in
        setTimeout(() => {
            notification.classList.add('show');
        }, 10);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                notification.remove();
            }, 300);
        }, 5000);
    }

    // Add notification styling
    const style = document.createElement('style');
    style.textContent = `
        .notification {
            position: fixed;
            top: 20px;
            right: 20px;
            background-color: var(--color-bg-secondary);
            border-left: 4px solid var(--color-accent-success);
            padding: 15px 20px;
            border-radius: var(--radius-sm);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
            z-index: 1000;
            max-width: 350px;
            transform: translateX(400px);
            opacity: 0;
            transition: transform 0.3s, opacity 0.3s;
        }
        
        .notification.show {
            transform: translateX(0);
            opacity: 1;
        }
        
        .notification-title {
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .notification-message {
            font-size: 0.9rem;
            color: var(--color-text-secondary);
        }
        
        .notification-close {
            position: absolute;
            top: 10px;
            right: 10px;
            background: none;
            border: none;
            color: var(--color-text-muted);
            font-size: 16px;
            cursor: pointer;
        }
        
        .notification-close:hover {
            color: var(--color-text-primary);
        }
    `;
    document.head.appendChild(style);
});