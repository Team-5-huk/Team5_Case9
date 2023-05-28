import { makeAutoObservable } from 'mobx';

class linkStore {
    linkLogin = 'http://87.244.7.150:8000/api/user/login/';
	linkGetUser = 'http://87.244.7.150:8000/api/user/me/';
	linkGetProjects = 'http://87.244.7.150:8000/api/samolet/projects/';
	linkGetSections = 'http://87.244.7.150:8000/api/samolet/buildings/';
	linkGetFlats = 'http://87.244.7.150:8000/api/samolet/sections/';
	linkCreateVideo = 'http://87.244.7.150:8000/api/samolet/checks/';

	constructor() {
		makeAutoObservable(this);
	}
}

export default new linkStore();