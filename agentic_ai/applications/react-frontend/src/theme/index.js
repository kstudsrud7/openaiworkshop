import { createTheme } from '@mui/material';

/**
 * Application theme configuration
 * Defines the color palette, typography, and other theme settings
 */
export const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#0c72b6',
      dark: '#085a91',
      light: '#5aa6d6',
    },
    secondary: {
      main: '#1e293b',
    },
    info: {
      main: '#0c72b6',
    },
    success: {
      main: '#2f855a',
    },
    warning: {
      main: '#b7791f',
    },
    text: {
      primary: '#0f172a',
      secondary: '#334155',
    },
    background: {
      default: '#f8fafc',
      paper: '#ffffff',
    },
    divider: '#e2e8f0',
  },
  shape: {
    borderRadius: 12,
  },
  typography: {
    fontFamily: 'Manrope, Inter, Segoe UI, Roboto, Helvetica, Arial, sans-serif',
    h5: {
      fontWeight: 700,
      letterSpacing: '-0.01em',
    },
    h6: {
      fontWeight: 700,
    },
    button: {
      textTransform: 'none',
      fontWeight: 600,
    },
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        '@import':
          "url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap')",
        body: {
          backgroundColor: '#f8fafc',
          color: '#0f172a',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          border: '1px solid #e2e8f0',
          boxShadow: '0 8px 28px rgba(15, 23, 42, 0.06)',
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: '#0f172a',
          backgroundImage:
            'linear-gradient(90deg, rgba(15,23,42,1) 0%, rgba(30,41,59,1) 70%, rgba(12,114,182,0.95) 100%)',
        },
      },
    },
    MuiAccordion: {
      styleOverrides: {
        root: {
          border: '1px solid #e2e8f0',
          borderRadius: 10,
          overflow: 'hidden',
          boxShadow: 'none',
          '&:before': {
            display: 'none',
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          fontWeight: 600,
        },
      },
    },
  },
});

/**
 * Event type colors for orchestrator events
 */
export const eventColors = {
  instruction: { bgColor: '#eff6ff', color: 'secondary' },
  task_ledger: { bgColor: '#e0f2fe', color: 'info' },
  user_task: { bgColor: '#f8fafc', color: 'default' },
  notice: { bgColor: '#fefce8', color: 'warning' },
  plan: { bgColor: '#e0f2fe', color: 'primary' },
  progress: { bgColor: '#eef2ff', color: 'info' },
  result: { bgColor: '#ecfdf3', color: 'success' },
  default: { bgColor: '#f8fafc', color: 'default' },
};

/**
 * Message background colors
 */
export const messageColors = {
  user: '#e0f2fe',
  assistant: '#ffffff',
  error: '#fef2f2',
};
