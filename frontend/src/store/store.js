import { makeAutoObservable } from 'mobx';
import a from '../assets/a.jpg';
import b from '../assets/b.jpg';
import c from '../assets/c.jpg';
import e from '../assets/e.jpg';

class store {
	authUser = '';
	files = [];
	sections = [];
	apartments = [];

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

	frames = [
		{
			id: 1,
			name: 'Корпус 1',
			isShow: true,
			info: 'info1',
			sections: ['Секция 1', 'Секция 2', 'Секция 3', 'Секция 4', 'Секция 5'],
		},
		{
			id: 2,
			name: 'Корпус 2',
			isShow: false,
			info: 'info2',
			sections: ['Секция 1', 'Секция 2', 'Секция 3', 'Секция 4', 'Секция 5'],
		},
		{
			id: 3,
			name: 'Корпус 3',
			isShow: false,
			info: 'info3',
			sections: ['Секция 1', 'Секция 2', 'Секция 3', 'Секция 4', 'Секция 5'],
		},
		{
			id: 4,
			name: 'Корпус 4',
			isShow: false,
			info: 'info4',
			sections: ['Секция 1', 'Секция 2', 'Секция 3', 'Секция 4', 'Секция 5'],
		},
	];

	constructor() {
		makeAutoObservable(this);

		this.authUser = localStorage.getItem('role') || '';
	}

	setApartments = (apartments) => {
		this.apartments = apartments;
	};

	setSections = (sections) => {
		this.sections = sections;
	};

	setFiles = (files) => {
		this.files = files;
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
