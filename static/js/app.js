import { isAuthenticated, signOut, getCurrentUser } from './auth.js';

// Initialize application
async function initApp() {
    const authenticated = await isAuthenticated();
    
    // Get navigation elements
    const loginButton = document.getElementById('login-button');
    const signupButton = document.getElementById('signup-button');
    const logoutButton = document.getElementById('logout-button');
    const userProfileButton = document.getElementById('user-profile');
    const userProfileContainer = document.getElementById('user-profile-container');
    
    if (authenticated) {
        // User is logged in
        if (loginButton) loginButton.style.display = 'none';
        if (signupButton) signupButton.style.display = 'none';
        if (logoutButton) logoutButton.style.display = 'block';
        if (userProfileContainer) userProfileContainer.style.display = 'block';
        
        // Display user info
        const user = await getCurrentUser();
        if (user && userProfileButton) {
            userProfileButton.textContent = user.email;
        }
    } else {
        // User is not logged in
        if (loginButton) loginButton.style.display = 'block';
        if (signupButton) signupButton.style.display = 'block';
        if (logoutButton) logoutButton.style.display = 'none';
        if (userProfileContainer) userProfileContainer.style.display = 'none';
    }
    
    // Add logout handler
    if (logoutButton) {
        logoutButton.addEventListener('click', async (e) => {
            e.preventDefault();
            const success = await signOut();
            if (success) {
                window.location.href = '/login';
            }
        });
    }
}

// Run initialization
document.addEventListener('DOMContentLoaded', initApp); 