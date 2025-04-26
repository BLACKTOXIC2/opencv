import supabase from './supabase.js';

// Check if user is authenticated
export async function isAuthenticated() {
    const { data, error } = await supabase.auth.getSession();
    if (error) {
        console.error('Error checking authentication:', error);
        return false;
    }
    return data.session !== null;
}

// Get current user
export async function getCurrentUser() {
    const { data, error } = await supabase.auth.getUser();
    if (error) {
        console.error('Error getting user:', error);
        return null;
    }
    return data?.user || null;
}

// Sign out
export async function signOut() {
    const { error } = await supabase.auth.signOut();
    if (error) {
        console.error('Error signing out:', error);
        return false;
    }
    
    // Clear any auth cookies on sign out
    document.cookie = "sb-auth-token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";
    
    return true;
}

// Protect routes that require authentication
export async function protectRoute() {
    const authenticated = await isAuthenticated();
    if (!authenticated) {
        // Redirect to login and include the current URL as the redirect target
        const currentPath = window.location.pathname;
        window.location.href = `/login?redirect=${encodeURIComponent(currentPath)}`;
        return false;
    }
    return true;
}

// Listen for auth state changes
export function onAuthStateChange(callback) {
    return supabase.auth.onAuthStateChange((event, session) => {
        // When user signs in, set auth cookie for server-side auth check
        if (event === 'SIGNED_IN' && session) {
            // Set a cookie that server can check
            document.cookie = `sb-auth-token=${session.access_token}; path=/; max-age=${session.expires_in}`;
        }
        
        // When user signs out, remove the cookie
        if (event === 'SIGNED_OUT') {
            document.cookie = "sb-auth-token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";
        }
        
        callback(event, session);
    });
}

// Initialize auth
export async function initAuth() {
    // Check URL for access_token fragment (from OAuth or magic link)
    if (window.location.hash) {
        const { data, error } = await supabase.auth.getSession();
        if (data.session) {
            // Set auth cookie when session exists
            document.cookie = `sb-auth-token=${data.session.access_token}; path=/; max-age=${data.session.expires_in}`;
            
            // Redirect to home after successful auth
            window.location.href = '/';
        }
    }
    
    // Set up auth state change listener
    onAuthStateChange((event, session) => {
        console.log('Auth state changed:', event);
    });
} 