# Supabase Authentication Setup

This project uses Supabase for authentication. Follow these steps to set up authentication:

## 1. Create a Supabase Account

- Go to [https://supabase.com/](https://supabase.com/) and sign up for a free account
- Create a new project

## 2. Set Up Authentication

1. Go to your Supabase dashboard > Authentication > Providers
2. Make sure "Email" provider is enabled
3. Configure email templates in the "Email Templates" section

## 3. Configure Your Application

1. In your project, open the file `static/js/supabase.js`
2. Replace the placeholders with your Supabase URL and anon key:

```javascript
const supabaseUrl = 'YOUR_SUPABASE_URL';
const supabaseKey = 'YOUR_SUPABASE_ANON_KEY';
```

You can find these values in the Supabase dashboard under Settings > API.

## 4. Configure Site URL

1. In your Supabase dashboard, go to Authentication > URL Configuration
2. Add your site URL (e.g., `http://localhost:8000` for local development)

## 5. Test the Authentication

- Try signing up with a new account
- Try logging in with the created account
- Test the "Forgot Password" flow

## Additional Configuration Options

### Email Confirmation

By default, Supabase requires email confirmation. You can change this in:
- Authentication > Email Templates > Confirm signup
- Enable/disable "Enable email confirmations"

### Password Strength

You can adjust password requirements in:
- Authentication > Policies

### Social Logins

For social login integration:
1. Authentication > Providers
2. Enable and configure the desired providers (Google, GitHub, etc.)
3. Follow the provider-specific setup instructions

## Troubleshooting

- Check browser console for error messages
- Verify your Supabase URL and anon key are correct
- Ensure your site URL is properly configured in Supabase
- Check Supabase logs under "Authentication > Activity" 