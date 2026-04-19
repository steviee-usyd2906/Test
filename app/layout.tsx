import './globals.css';
import Sidebar from './Sidebar';

export const metadata = {
  title: 'YouRisk | Media Valuation',
  description: 'The enterprise standard for digital media asset valuation.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className="min-h-screen font-sans flex text-slate-900"
        style={{ background: 'linear-gradient(145deg, #fdfbf7 0%, #f0ede6 40%, #e2e8f0 100%)' }}
      >
        {/* Soft orb decorations */}
        <div className="fixed inset-0 pointer-events-none overflow-hidden" style={{ zIndex: 0 }}>
          <div style={{ position: 'absolute', top: '-10%', left: '-5%', width: '420px', height: '420px', borderRadius: '50%', background: 'radial-gradient(circle, rgba(37, 99, 235, 0.12) 0%, transparent 70%)' }} />
          <div style={{ position: 'absolute', bottom: '5%', right: '-8%', width: '500px', height: '500px', borderRadius: '50%', background: 'radial-gradient(circle, rgba(217, 204, 185, 0.3) 0%, transparent 70%)' }} />
        </div>

        <Sidebar />

        {/* MAIN CONTENT AREA */}
        <main className="flex-1 p-10 overflow-y-auto relative" style={{ zIndex: 1 }}>
          <div className="max-w-4xl mx-auto space-y-6">
            {children}
          </div>
        </main>
      </body>
    </html>
  );
}
