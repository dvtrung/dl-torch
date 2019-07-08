// @flow
import type { GetState, Dispatch } from '../reducers/types';
import {getLogPath, sshConnect} from "../utils/ssh";
import {ipcRenderer} from "electron";

const path = require('path');
const fs = require('fs');

export const LOAD_DIRECTORY = 'LOAD_DIRECTORY';
export const SELECT_MODEL = 'SELECT_MODEL';
export const SSH_CONNECT = 'SSH_CONNECT';
export const SYNC_EPOCH_STATS_START = 'SYNC_EPOCH_STATS_START';
export const SYNC_EPOCH_STATS_END = 'SYNC_EPOCH_STATS_END';
export const SYNC_EPOCH_STEP_STATS_UPDATE = 'SYNC_EPOCH_STEP_STATS_UPDATE';
export const SYNC_EPOCH_STEP_STATS_START = 'SYNC_EPOCH_STEP_STATS_START';

export function loadModelsAsync(path) {
  return (dispatch: Dispatch) => {
    fs.readdir(`${path}\\model_configs`, function (err, filenames) {
      if (err) {
          return console.log('Unable to scan directory: ' + err);
      }
      
      const models = {};
      filenames.forEach(filename => {
        const name = filename.substr(0, filename.length - 4)
        models[name] = {
          path: filename
        };
      });

      dispatch({
        type: LOAD_DIRECTORY,
        name: path,
        path,
        models
      });
    });
  }
}

export function onModelSelected(key) {
  return (dispatch: Dispatch, getState) => {
    // const content = fs.readFileSync(
    //   path.join(root, 'model_configs', `${modelName}.yml`), { encoding: 'utf-8' });
    const state = getState();
    const model = state.home.models[key];
    const machine = state.home.machines[model.machine]
    dispatch({
      type: SELECT_MODEL,
      key,
      // config: content
    });
    syncEpochStats(model, machine)(dispatch);
    syncEpochStepStats(model, machine)(dispatch);
  }
}

export function fetchModelLog(machine, model) {
  sshConnect(machine, (client) => {

  })
}

export function streamModelLog(machine, model) {
  sshConnect(machine, (client) => {
    client.exec(`logs/${model.name}/\$(ls logs/${model.name}/ -S | tail -n 1)/results.json`, (err, stream) => {
      if (err) throw err;
    });
    client.exec('tail -f ~/output.log', function(err, stream) {
      if (err) throw err;
      stream.on('close', function(code, signal) {
        console.log('Stream :: close :: code: ' + code + ', signal: ' + signal);
        client.end();
      }).on('data', function(data) {
        console.log('STDOUT: ' + data);
      }).stderr.on('data', function(data) {
        console.log('STDERR: ' + data);
      });
    });
  })
}

export function syncEpochStats(model, machine, force = false) {
  return (dispatch: Dispatch) => {
    if (!force && model.stats.lastUpdated
      && new Date() - model.stats.lastUpdated < 1000 * 60) return;
    dispatch({ type: SYNC_EPOCH_STATS_START, model });
    let ret = "";
    sshConnect(machine, (client) => {
      client.exec(`tail -n 100 ${getLogPath(model)}/epoch-info.log`, (err, stream) => {
        if (err) throw err;
        stream.on('close', () => {
          client.end();
          ret = ret.split('\n')
            .filter((line: string) => line.trim() !== "")
            .map((line) => JSON.parse(line));
          dispatch({ type: SYNC_EPOCH_STATS_END, model, data: ret })
        }).on('data', (data) => {
          ret += data
        });
      });
    }, (err) => {
      ipcRenderer.send("error-box", {
        message: err.toString()
      });
    });
  }
}

export function syncEpochStepStats(model, machine) {
  return (dispatch: Dispatch) => {
    let ret = "";
    dispatch({ type: SYNC_EPOCH_STEP_STATS_START, model });
    sshConnect(machine, (client) => {
      client.exec(`tail -f -n 100 ${getLogPath(model)}/epoch-step-info.log`, (err, stream) => {
        if (err) throw err;
        stream.on('data', (data) => {
          ret += data;
          const lines = ret.split('\n');
          lines.slice(0, ret.length - 1).forEach((line: string) => {
            try {
              dispatch({type: SYNC_EPOCH_STEP_STATS_UPDATE, model, data: JSON.parse(line)})
            } catch (e) {

            }
          });
          ret = ret[ret.length - 1];
        });
      });
    }, (err) => {
      ipcRenderer.send("error-box", {
        message: err.toString()
      });
    });
  }
}
