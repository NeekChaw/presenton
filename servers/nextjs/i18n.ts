import {getRequestConfig} from 'next-intl/server';

export default getRequestConfig(async ({locale}) => {
  console.log('Received locale:', locale);

  // Provide a default locale if none is provided
  const activeLocale = locale || 'en';

  try {
    const messages = (await import(`./messages/${activeLocale}.json`)).default;
    return {
      messages
    };
  } catch (error) {
    console.error('Error loading messages for locale:', activeLocale, error);
    // Fallback to en.json
    const messages = (await import(`./messages/en.json`)).default;
    return {
      messages
    };
  }
});
