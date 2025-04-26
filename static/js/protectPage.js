import { protectRoute, getCurrentUser } from './auth.js';

// Check if user is authenticated and redirect to login if not
(async function() {
    const isAuth = await protectRoute();
    
    if (isAuth) {
        // User is authenticated, get user details
        const user = await getCurrentUser();
        
        // Display user information if available
        if (user) {
            const userDisplayElement = document.getElementById('user-display');
            if (userDisplayElement) {
                userDisplayElement.textContent = user.email;
            }
            
            // Also store the user data for other scripts to access
            window.currentUser = user;
        }
    }
})(); 