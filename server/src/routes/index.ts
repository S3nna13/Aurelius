import type { Router } from 'express';
import { health } from './health.js';
import { status, agentDetail, updateAgent } from './status.js';
import { listModels, getModel, setModelState, getModelStats } from './models.js';
import { listTrainingRuns, getTrainingRun, createTrainingRun, trainingStats } from './training.js';
import { command } from './command.js';
import { activity } from './activity.js';
import { listNotifications, notificationStats, createNotification, markNotificationRead, markAllNotificationsRead, getNotifPrefs, setNotifPrefs } from './notifications.js';
import { listSkills, skillDetail, executeSkill } from './skills.js';
import { listWorkflows, workflowDetail, triggerWorkflow } from './workflows.js';
import { memoryLayers, memoryEntries, addMemoryEntry } from './memory.js';
import { getConfig, setConfig } from './config.js';
import { listModes } from './modes.js';
import { getLogs } from './logs.js';
import { validateLicense, activateLicense } from './license.js';
import { proxyToPython } from './proxy.js';

export function registerRoutes(router: Router): void {
  router.get('/health', health);
  router.get('/status', status);
  router.get('/activity', activity);

  router.get('/notifications', listNotifications);
  router.get('/notifications/stats', notificationStats);
  router.post('/notifications', createNotification);
  router.post('/notifications/read', markNotificationRead);
  router.post('/notifications/read-all', markAllNotificationsRead);
  router.get('/notifications/preferences', getNotifPrefs);
  router.post('/notifications/preferences', setNotifPrefs);

  router.get('/skills', listSkills);
  router.get('/skills/:id', skillDetail);
  router.post('/skills/execute', executeSkill);

  router.get('/workflows', listWorkflows);
  router.get('/workflows/:id', workflowDetail);
  router.post('/workflows/:id/trigger', triggerWorkflow);

  router.get('/memory', memoryLayers);
  router.get('/memory/layers', memoryLayers);
  router.get('/memory/entries', memoryEntries);
  router.post('/memory/entries', addMemoryEntry);

  router.get('/config', getConfig);
  router.post('/config', setConfig);

  router.get('/modes', listModes);

  router.get('/logs', getLogs);

  router.get('/license/validate', validateLicense);
  router.post('/license/activate', activateLicense);

  router.get('/agents/:id', agentDetail);
  router.post('/agents/:id/state', updateAgent);

  router.get('/models', listModels);
  router.get('/models/stats', getModelStats);
  router.get('/models/:id', getModel);
  router.post('/models/:id/state', setModelState);

  router.get('/training', listTrainingRuns);
  router.get('/training/stats', trainingStats);
  router.get('/training/:id', getTrainingRun);
  router.post('/training', createTrainingRun);
  router.post('/command', command);

  router.use('/chat', proxyToPython);
}
