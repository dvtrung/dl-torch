// @flow
import React, { Component } from 'react';
import Home from '../components/Home';
import { bindActionCreators } from 'redux';
import { connect } from 'react-redux';
import * as HomeActions from '../actions/home';
import * as SettingActions from '../actions/settings';

function mapStateToProps(state) {
  return {
    settings: state.settings,
    machines: state.home.machines,
    dirs: state.home.dirs,
    currentDir: state.home.currentDir,
    selectedModelKey: state.home.selectedModelKey,
    selectedMachineKey: state.home.selectedMachineKey
  };
}

function mapDispatchToProps(dispatch) {
  return bindActionCreators({
    ...HomeActions, 
    ...SettingActions
  }, dispatch)
}

export default connect(
  mapStateToProps,
  mapDispatchToProps
)(Home);
