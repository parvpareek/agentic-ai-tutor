import React, { useState, useEffect } from 'react';
import { getLessonSessions, getLessonSession, completeLessonSession, LessonSession, DetailedLessonSession } from '../api/tutorApi';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import { Button } from '../components/ui/button';
import { ScrollArea } from '../components/ui/scroll-area';
import { Separator } from '../components/ui/separator';
import { Calendar, Clock, FileText, MessageSquare, CheckCircle, Circle, AlertCircle } from 'lucide-react';
import { format } from 'date-fns';

const History: React.FC = () => {
  const [sessions, setSessions] = useState<LessonSession[]>([]);
  const [selectedSession, setSelectedSession] = useState<DetailedLessonSession | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadSessions();
  }, []);

  const loadSessions = async () => {
    try {
      setLoading(true);
      const data = await getLessonSessions();
      setSessions(data);
    } catch (err) {
      setError('Failed to load lesson history');
      console.error('Error loading sessions:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadSessionDetails = async (sessionId: string) => {
    try {
      const session = await getLessonSession(sessionId);
      setSelectedSession(session);
    } catch (err) {
      setError('Failed to load session details');
      console.error('Error loading session details:', err);
    }
  };

  const handleCompleteSession = async (sessionId: string) => {
    try {
      await completeLessonSession(sessionId);
      await loadSessions(); // Refresh the list
      if (selectedSession?.session_id === sessionId) {
        setSelectedSession(null);
      }
    } catch (err) {
      setError('Failed to complete session');
      console.error('Error completing session:', err);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'active':
        return <Circle className="h-4 w-4 text-blue-500" />;
      default:
        return <AlertCircle className="h-4 w-4 text-gray-500" />;
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'completed':
        return <Badge variant="default" className="bg-green-100 text-green-800">Completed</Badge>;
      case 'active':
        return <Badge variant="secondary" className="bg-blue-100 text-blue-800">Active</Badge>;
      default:
        return <Badge variant="outline">Unknown</Badge>;
    }
  };

  if (loading) {
    return (
      <div className="container mx-auto p-6">
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p className="text-gray-600">Loading lesson history...</p>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container mx-auto p-6">
        <div className="text-center">
          <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-900 mb-2">Error</h2>
          <p className="text-gray-600 mb-4">{error}</p>
          <Button onClick={loadSessions}>Try Again</Button>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Lesson History</h1>
        <p className="text-gray-600">Review your past learning sessions and progress</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Sessions List */}
        <div>
          <h2 className="text-xl font-semibold text-gray-900 mb-4">All Sessions</h2>
          <ScrollArea className="h-[600px]">
            <div className="space-y-4">
              {sessions.length === 0 ? (
                <Card>
                  <CardContent className="p-6 text-center">
                    <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                    <h3 className="text-lg font-medium text-gray-900 mb-2">No Sessions Yet</h3>
                    <p className="text-gray-600">Start learning by uploading a document!</p>
                  </CardContent>
                </Card>
              ) : (
                sessions.map((session) => (
                  <Card 
                    key={session.session_id}
                    className={`cursor-pointer transition-all hover:shadow-md ${
                      selectedSession?.session_id === session.session_id ? 'ring-2 ring-blue-500' : ''
                    }`}
                    onClick={() => loadSessionDetails(session.session_id)}
                  >
                    <CardHeader className="pb-3">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <CardTitle className="text-lg flex items-center gap-2">
                            {getStatusIcon(session.status)}
                            {session.document_name}
                          </CardTitle>
                          <CardDescription className="mt-1">
                            {session.document_type.toUpperCase()} • Session {session.session_id.split('_')[1]}
                          </CardDescription>
                        </div>
                        {getStatusBadge(session.status)}
                      </div>
                    </CardHeader>
                    <CardContent className="pt-0">
                      <div className="flex items-center gap-4 text-sm text-gray-600">
                        <div className="flex items-center gap-1">
                          <Calendar className="h-4 w-4" />
                          {format(new Date(session.created_at), 'MMM dd, yyyy')}
                        </div>
                        {session.completed_at && (
                          <div className="flex items-center gap-1">
                            <Clock className="h-4 w-4" />
                            Completed {format(new Date(session.completed_at), 'MMM dd')}
                          </div>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                ))
              )}
            </div>
          </ScrollArea>
        </div>

        {/* Session Details */}
        <div>
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Session Details</h2>
          {selectedSession ? (
            <ScrollArea className="h-[600px]">
              <Card>
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div>
                      <CardTitle className="text-lg flex items-center gap-2">
                        {getStatusIcon(selectedSession.status)}
                        {selectedSession.document_name}
                      </CardTitle>
                      <CardDescription className="mt-1">
                        {selectedSession.document_type.toUpperCase()} • {selectedSession.concepts.length} concepts
                      </CardDescription>
                    </div>
                    {getStatusBadge(selectedSession.status)}
                  </div>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* Session Info */}
                  <div>
                    <h3 className="font-medium text-gray-900 mb-2">Session Information</h3>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-gray-600">Created:</span>
                        <p className="font-medium">{format(new Date(selectedSession.created_at), 'MMM dd, yyyy HH:mm')}</p>
                      </div>
                      {selectedSession.completed_at && (
                        <div>
                          <span className="text-gray-600">Completed:</span>
                          <p className="font-medium">{format(new Date(selectedSession.completed_at), 'MMM dd, yyyy HH:mm')}</p>
                        </div>
                      )}
                    </div>
                  </div>

                  <Separator />

                  {/* Concepts */}
                  <div>
                    <h3 className="font-medium text-gray-900 mb-2">Concepts Covered</h3>
                    <div className="flex flex-wrap gap-2">
                      {selectedSession.concepts.map((concept, index) => (
                        <Badge key={index} variant="outline">
                          {concept.label}
                        </Badge>
                      ))}
                    </div>
                  </div>

                  <Separator />

                  {/* Study Plan */}
                  <div>
                    <h3 className="font-medium text-gray-900 mb-2">Study Plan</h3>
                    <div className="space-y-2">
                      {selectedSession.study_plan.map((step, index) => (
                        <div key={index} className="flex items-center gap-2 text-sm">
                          <div className="w-6 h-6 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center text-xs font-medium">
                            {index + 1}
                          </div>
                          <span className="capitalize">{step.action.replace('_', ' ')}</span>
                          <span className="text-gray-600">- {step.topic}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <Separator />

                  {/* Quiz Results */}
                  {selectedSession.quiz_results && selectedSession.quiz_results.length > 0 && (
                    <div>
                      <h3 className="font-medium text-gray-900 mb-2">Quiz Results</h3>
                      <div className="space-y-2">
                        {selectedSession.quiz_results.map((result, index) => (
                          <div key={index} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                            <span className="text-sm">{result.topic || `Quiz ${index + 1}`}</span>
                            <Badge variant={result.score_percentage >= 70 ? "default" : "destructive"}>
                              {result.score_percentage?.toFixed(1)}%
                            </Badge>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  <Separator />

                  {/* Conversation History */}
                  {selectedSession.detailed_messages && selectedSession.detailed_messages.length > 0 && (
                    <div>
                      <h3 className="font-medium text-gray-900 mb-2 flex items-center gap-2">
                        <MessageSquare className="h-4 w-4" />
                        Conversation History
                      </h3>
                      <div className="space-y-3 max-h-48 overflow-y-auto">
                        {selectedSession.detailed_messages.map((message, index) => (
                          <div key={index} className={`p-3 rounded-lg ${
                            message.role === 'tutor' 
                              ? 'bg-blue-50 border-l-4 border-blue-200' 
                              : 'bg-gray-50 border-l-4 border-gray-200'
                          }`}>
                            <div className="flex items-center gap-2 mb-1">
                              <span className="text-xs font-medium text-gray-600">
                                {message.role === 'tutor' ? 'Tutor' : 'You'}
                              </span>
                              <span className="text-xs text-gray-500">
                                {format(new Date(message.timestamp), 'HH:mm')}
                              </span>
                            </div>
                            <p className="text-sm text-gray-800">{message.content}</p>
                            {message.sources && message.sources.length > 0 && (
                              <div className="mt-2">
                                <p className="text-xs text-gray-600">Sources:</p>
                                <div className="flex flex-wrap gap-1 mt-1">
                                  {message.sources.map((source, idx) => (
                                    <Badge key={idx} variant="outline" className="text-xs">
                                      {source}
                                    </Badge>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Actions */}
                  {selectedSession.status === 'active' && (
                    <div className="pt-4">
                      <Button 
                        onClick={() => handleCompleteSession(selectedSession.session_id)}
                        className="w-full"
                      >
                        Mark as Completed
                      </Button>
                    </div>
                  )}
                </CardContent>
              </Card>
            </ScrollArea>
          ) : (
            <Card>
              <CardContent className="p-6 text-center">
                <MessageSquare className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">Select a Session</h3>
                <p className="text-gray-600">Choose a session from the list to view details</p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};

export default History;

