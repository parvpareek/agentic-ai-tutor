import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { TutorProvider } from "./context/TutorContext";
import Navigation from "./components/Navigation";
import Upload from "./pages/Upload";
import Concepts from "./pages/Concepts";
import Learn from "./pages/Learn";
import Quiz from "./pages/Quiz";
import Results from "./pages/Results";
import History from "./pages/History";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <TutorProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <Navigation />
          <Routes>
            <Route path="/" element={<Upload />} />
            <Route path="/concepts" element={<Concepts />} />
            <Route path="/learn" element={<Learn />} />
            <Route path="/quiz" element={<Quiz />} />
            <Route path="/results" element={<Results />} />
            <Route path="/history" element={<History />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </TutorProvider>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
