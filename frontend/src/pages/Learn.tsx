// Learning flow disabled - redirect to concepts page
import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

export default function Learn() {
  const navigate = useNavigate();
  
  useEffect(() => {
    navigate('/concepts', { replace: true });
  }, [navigate]);
  
  return null;
}
