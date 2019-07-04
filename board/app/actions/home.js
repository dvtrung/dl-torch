// @flow
import type { GetState, Dispatch } from '../reducers/types';
import {getLogPath, sshConnect} from "../utils/ssh";
import {ipcRenderer} from "electron";

const path = require('path');
const fs = require('fs');

export const LOAD_DIRECTORY = 'LOAD_DIRECTORY';
export const SELECT_MODEL = 'SELECT_MODEL';
export const SSH_CONNECT = 'SSH_CONNECT';
export const SYNC_TRAINING_STATS_START = 'SYNC_TRAINING_STATS_START';
export const SYNC_TRAINING_STATS_END = 'SYNC_TRAINING_STATS_END';

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

export function onModelSelected(root, modelName) {
  return (dispatch: Dispatch) => {
    const content = fs.readFileSync(
      path.join(root, 'model_configs', `${modelName}.yml`), 
      { encoding: 'utf-8' });
    dispatch({
      type: SELECT_MODEL,
      model: {
        name: modelName, root,
      },
      config: content
    });
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

export function syncTrainingStats(model, machine) {
  return (dispatch: Dispatch) => {
    console.log("start");
    dispatch({ type: SYNC_TRAINING_STATS_START, model });
    let ret = "";
    sshConnect(machine, (client) => {
      console.log(`tail -n 1 ${getLogPath(model)}/epoch-info.log`)
      client.exec(`tail -n 1 ${getLogPath(model)}/epoch-info.log`, (err, stream) => {
        if (err) throw err;
        stream.on('close', () => {
          client.end();
          dispatch({ type: SYNC_TRAINING_STATS_END, model, data: JSON.parse(ret) })
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
