/**
 * Application configuration
 * 
 * This file contains configuration settings for the application.
 * Update these values with your actual Supabase project credentials.
 */

export const SUPABASE_CONFIG = {
    // Replace with your Supabase project URL (from Project Settings > API)
    URL: 'https://dahqxjnokapjgrxkbrdo.supabase.co',
    
    // Replace with your Supabase anon/public key (from Project Settings > API)
    ANON_KEY: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRhaHF4am5va2FwamdyeGticmRvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDQ5Njc0OTksImV4cCI6MjA2MDU0MzQ5OX0.b8Wy_3lI-_d6oI_GIVBNsdltq6a0hNszXOFm5SeeT3U',
};

// For development use
export const IS_DEVELOPMENT = window.location.hostname === 'localhost' || 
                            window.location.hostname === '127.0.0.1'; 