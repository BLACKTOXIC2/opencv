// Import Supabase from CDN
import { createClient } from 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2/+esm';
import { SUPABASE_CONFIG, IS_DEVELOPMENT } from './config.js';

// Initialize Supabase client with values from config
const supabaseUrl = SUPABASE_CONFIG.URL;
const supabaseKey = SUPABASE_CONFIG.ANON_KEY;

// Warning for default credentials
if (supabaseUrl === 'https://dahqxjnokapjgrxkbrdo.supabase.co' || supabaseKey === 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRhaHF4am5va2FwamdyeGticmRvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDQ5Njc0OTksImV4cCI6MjA2MDU0MzQ5OX0.b8Wy_3lI-_d6oI_GIVBNsdltq6a0hNszXOFm5SeeT3U') {
    console.warn('⚠️ Default Supabase credentials detected! Please update your credentials in static/js/config.js');
    
    if (IS_DEVELOPMENT) {
        // Show warning in development mode
        const warningElem = document.createElement('div');
        warningElem.style.cssText = 'position:fixed; top:0; left:0; right:0; background-color:#ffdd57; color:#000; padding:10px; text-align:center; z-index:9999; font-family:sans-serif';
        warningElem.innerHTML = '⚠️ <strong>Default Supabase credentials</strong>: Update your credentials in <code>static/js/config.js</code>';
        document.body.appendChild(warningElem);
    }
}

// Create the client
let supabase;
try {
    supabase = createClient(supabaseUrl, supabaseKey);
} catch (error) {
    console.error('Error initializing Supabase client:', error);
    // Create an empty mock client for graceful degradation
    supabase = {
        auth: {
            signInWithPassword: () => Promise.resolve({ data: null, error: { message: 'Supabase not configured' } }),
            signUp: () => Promise.resolve({ data: null, error: { message: 'Supabase not configured' } }),
            resetPasswordForEmail: () => Promise.resolve({ data: null, error: { message: 'Supabase not configured' } }),
            updateUser: () => Promise.resolve({ data: null, error: { message: 'Supabase not configured' } }),
            getSession: () => Promise.resolve({ data: { session: null }, error: null }),
            getUser: () => Promise.resolve({ data: { user: null }, error: null }),
            signOut: () => Promise.resolve({ error: null }),
            onAuthStateChange: (callback) => { callback('SIGNED_OUT', null); return { data: { subscription: { unsubscribe: () => {} } } }; }
        }
    };
}

export default supabase; 