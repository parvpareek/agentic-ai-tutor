import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Button } from './ui/button';
import { Home, BookOpen, History, Upload } from 'lucide-react';

const Navigation: React.FC = () => {
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Upload', icon: Upload },
    { path: '/concepts', label: 'Concepts', icon: BookOpen },
    { path: '/learn', label: 'Learn', icon: BookOpen },
    { path: '/quiz', label: 'Quiz', icon: BookOpen },
    { path: '/results', label: 'Results', icon: BookOpen },
    { path: '/history', label: 'History', icon: History },
  ];

  return (
    <nav className="bg-white border-b border-gray-200 px-4 py-3">
      <div className="container mx-auto flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Home className="h-6 w-6 text-blue-600" />
          <span className="text-xl font-bold text-gray-900">AI Tutor</span>
        </div>
        
        <div className="flex items-center space-x-2">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;
            
            return (
              <Link key={item.path} to={item.path}>
                <Button
                  variant={isActive ? "default" : "ghost"}
                  size="sm"
                  className={`flex items-center space-x-2 ${
                    isActive 
                      ? 'bg-blue-600 text-white' 
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  <span>{item.label}</span>
                </Button>
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
};

export default Navigation;

