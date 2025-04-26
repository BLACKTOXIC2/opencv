# Supabase Authentication Setup Guide

## Overview

This application uses Supabase for authentication. Follow these steps to set up your own Supabase project and connect it to the application.

## Step 1: Create a Supabase Account and Project

1. Go to https://supabase.com/ and sign up for a free account
2. Once signed in, create a new project
3. Choose a name and password for your project

## Step 2: Get Your API Credentials

1. In your Supabase project dashboard, go to **Project Settings** (gear icon in the bottom left)
2. Click on **API** in the sidebar
3. You'll need two values:
   - **Project URL**: Shown at the top (looks like `https://xxxxxxxxxxxx.supabase.co`)
   - **anon (public) key**: A long string starting with "ey..."

## Step 3: Configure Your Application

1. Open the file `static/js/config.js`
2. Replace the placeholder values with your actual Supabase credentials:

```javascript
export const SUPABASE_CONFIG = {
    URL: 'https://your-project-id.supabase.co',  // Replace with your actual URL
    ANON_KEY: 'your-actual-anon-key',            // Replace with your actual key
};
```

## Step 4: Configure Supabase Authentication Settings

1. In your Supabase dashboard, go to **Authentication** > **Providers**
2. Make sure **Email** is enabled
3. Go to **URL Configuration**
4. Add your site URL (e.g., `http://localhost:8000` for local development)
5. In **Redirect URLs**, add:
   - `http://localhost:8000/login`
   - `http://localhost:8000/`

## Step 5: Email Templates (Optional)

1. In your Supabase dashboard, go to **Authentication** > **Email Templates**
2. Customize the templates as needed for:
   - Confirmation email
   - Invitation email
   - Magic link email
   - Reset password email

## Step 6: Testing Authentication

1. Start your application
2. Visit `/auth-test` to verify Supabase connection is working
3. Try signing up with a new account
4. Test login functionality
5. Test "Forgot Password" flow

## Troubleshooting

- **Error in the console**: Check that your URL and anon key are entered correctly
- **Redirect issues**: Verify your site URL is correctly configured in Supabase
- **Email not receiving**: Check spam folder or Supabase email logs
- **CORS errors**: Make sure your site URL is properly configured in Supabase

## Additional Resources

- [Supabase Auth Documentation](https://supabase.com/docs/guides/auth)
- [Supabase JavaScript Client](https://supabase.com/docs/reference/javascript/introduction) 