import { makeAutoObservable } from 'mobx';
import { makePersistable } from 'mobx-persist-store';

const getIsAuthorize = () => {
	const authData = localStorage.getItem('role');

	if (authData !== null) {
		return authData;
	}

	return 'logout';
};

class store {
	authUser = 'logout';

	sections = [];
	apartments = [];

	numberFlat = null;

	employees = [
		{
			id: 1,
			name: 'Алексей',
			lastName: 'Иванов',
			email: 'jonsnow@gmail.com',
			phone: '(665)121-5454',
			date: '03/12/2022',
		},
		{
			id: 2,
			name: 'Андрей',
			lastName: 'Иванов',
			email: 'cerseilannister@gmail.com',
			phone: '(421)314-2288',
			date: '06/15/2021',
		},
		{
			id: 3,
			name: 'Михаил',
			lastName: 'Иванов',
			email: 'jaimelannister@gmail.com',
			phone: '(422)982-6739',
			date: '05/02/2022',
		},
		{
			id: 4,
			name: 'Дмитрий',
			lastName: 'Иванов',
			email: 'anyastark@gmail.com',
			phone: '(921)425-6742',
			date: '03/21/2022',
		},
		{
			id: 5,
			name: 'Даниил',
			lastName: 'Иванов',
			email: 'daenerystargaryen@gmail.com',
			phone: '(421)445-1189',
			date: '01/12/2021',
		},
	];

	allObjects = [];

	frames = [];

	constructor() {
		makeAutoObservable(this);

		this.authUser = getIsAuthorize();

		makePersistable(this, {
			name: 'selectFlat',
			properties: ['numberFlat'],
			storage: window.localStorage,
		});
	}

	setApartments = (apartments) => {
		this.apartments = apartments;
	};

	setNumberFlat = (numberFlat) => {
		this.numberFlat = numberFlat;
	}

	setSections = (sections) => {
		this.sections = sections;
	};

	setAuthUser = (user) => {
		this.authUser = user;
	};

	setAllObjects = (objects = []) => {
		this.allObjects = objects;
	};

	setFrames = (frames = []) => {
		this.frames = frames;
	};

	setVisibleListFrame = (idFrame) => {
		this.frames.forEach((frame) => {
			frame.isShow = false;
		});

		const index = this.frames.findIndex(({ id }) => id === idFrame);
		if (index === -1) return;

		const { isShow } = this.frames[index];

		this.frames[index].isShow = !isShow;
	};
}

export default new store();
